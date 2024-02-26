import torch
import torchvision
from torch import nn, optim
from transformers import AutoTokenizer, CLIPTextModelWithProjection, CLIPVisionModelWithProjection
#from diffusers.optimization import get_scheduler

import sys
import os
from core_util import load_or_fail, setup_webdataset_path, MultiFilter, MultiGetter
from gdf_util import GDF, EpsilonTarget, CosineSchedule, VPScaler, CosineTNoiseCond, DDPMSampler, P2LossWeight, AdaptiveLossWeight
from model_util import EfficientNetEncoder, StageC, ResBlock, AttnBlock, TimestepBlock, FeedForwardBlock, Previewer
from dataset_util import BucketWalker
from bucketeer import Bucketeer
from fractions import Fraction

from torchtools.transforms import SmartCrop

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.utils.data import Dataset, DataLoader
from accelerate import init_empty_weights, Accelerator
from accelerate.utils import set_module_tensor_to_device
from contextlib import contextmanager
import yaml
import json
import numpy as np


# Handle special command line args
import argparse
parser = argparse.ArgumentParser(description="Simpler example of a Cascade training script.")
parser.add_argument("--yaml", default=None, type=str, help="The training configuration YAML")
args = parser.parse_args()

models = {}
settings = {}
info = {
	#"ema_loss": "",
	#"adaptive_loss": {}
}

def get_conditions(batch, models, extras):
	pass

def load_model(model, model_id=None, full_path=None, strict=True, settings=None):
	if model_id is not None and full_path is None:
		full_path = f"{settings['checkpoint_path']}/{settings['experiment_id']}/{model_id}.{settings['checkpoint_extension']}"
	elif full_path is None and model_id is None:
		raise ValueError("Loading a model expects full_path or model_id to be defined.")

	ckpt = load_or_fail(full_path, wandb_run_id=None)
	if ckpt is not None:
		model.load_state_dict(ckpt, strict=strict)
		del ckpt
	return model

# Replaced WarpCore with a more simplified version of it
# made compatible with HF Accelerate
def main():
	global settings
	global info
	global models
	# Basic Setup

	settings["checkpoint_extension"] = "safetensors"
	settings["allow_tf32"] = True
	settings["wandb_project"] = None
	settings["wandb_entity"] = None
	settings["clip_image_model_name"] = "openai/clip-vit-large-patch14"
	settings["clip_text_model_name"] = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
	
	gdf_loss_weight = P2LossWeight()
	if "adaptive_loss_weight" in settings:
		if settings["adaptive_loss_weight"]:
			gdf_loss_weight = AdaptiveLossWeight()
	settings["gdf"] = GDF(
		schedule=CosineSchedule(clamp_range=[0.0001, 0.9999]),
		input_scaler=VPScaler(), target=EpsilonTarget(),
		noise_cond=CosineTNoiseCond(),
		loss_weight=gdf_loss_weight,
	)

	settings["effnet_preprocess"] = torchvision.transforms.Compose([
		torchvision.transforms.Normalize(
			mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
		)
	])

	# Load config:
	loaded_config = ""
	if args.yaml is not None:
		if args.yaml.endswith(".yml") or args.yaml.endswith(".yaml"):
			with open(args.yaml, "r", encoding="utf-8") as file:
				loaded_config = yaml.safe_load(file)
		elif args.yaml.endswith(".json"):
			with open(args.yaml, "r", encoding="utf-8") as file:
				loaded_config = json.load(file)
		else:
			raise ValueError("Config file must either be a .yaml or .json file, stopping.")
		
		# Set things up
		settings = settings | loaded_config
	else:
		raise ValueError("No configuration supplied, stopping.")

	settings["clip_preprocess"] = torchvision.transforms.Compose([
		torchvision.transforms.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
		torchvision.transforms.CenterCrop(224),
		torchvision.transforms.Normalize(
			mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
		)
	])

	settings["transforms"] = torchvision.transforms.Compose([
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Resize(settings["image_size"], interpolation=torchvision.transforms.InterpolationMode.LANCZOS, antialias=True),
		SmartCrop(settings["image_size"], randomize_p=0.3, randomize_q=0.2)
	])


	full_path = f"{settings['checkpoint_path']}/{settings['experiment_id']}/info.json"
	info_dict = load_or_fail(full_path, wandb_run_id=None) or {}
	info = info | info_dict

	# Setup GDF buckets when resuming a training run
	if "adaptive_loss" in info:
		if "bucket_ranges" in info["adaptive_loss"] and "bucket_losses" in info["adaptive_loss"]:
			settings["gdf"].loss_weight.bucket_ranges = torch.tensor(info["adaptive_loss"]["bucket_ranges"])
			settings["gdf"].loss_weight.bucket_losses = torch.tensor(info["adaptive_loss"]["bucket_losses"])

	hf_accel_dtype = settings["dtype"]
	if settings["dtype"] == "bfloat16":
		hf_accel_dtype = "bf16"
	elif settings["dtype"] == "tf32":
		hf_accel_dtype = "no"
	elif settings["dtype"] != "fp16" or settings["dtype"] != "fp32":
		hf_accel_dtype = "no"
	
	accelerator = Accelerator(
		gradient_accumulation_steps=settings["grad_accum_steps"],
		mixed_precision=hf_accel_dtype,
		log_with="tensorboard",
		project_dir=f"{settings['output_path']}"
	)

	if settings["allow_tf32"]:
		torch.backends.cuda.matmul.allow_tf32 = True
		torch.backends.cudnn.allow_tf32 = True

	# Setup Dataloader:
	print("Loading Dataset[s].")
	pre_dataset = BucketWalker(
		reject_aspects=settings["reject_aspects"],
		transforms=torchvision.transforms.ToTensor(),
	)

	if "local_dataset_path" in settings:
		if type(settings["local_dataset_path"]) is list:
			for dir in settings["local_dataset_path"]:
				pre_dataset.scan_folder(dir)
		elif type(settings["local_dataset_path"]) is str:
			pre_dataset.scan_folder(settings["local_dataset_path"])
		else:
			raise ValueError("'local_dataset_path' must either be a string, or list of strings containing paths.")

	pre_dataset.bucketize(settings["batch_size"], settings["seed"])
	print(f"Total Invalid Files:  {pre_dataset.get_rejects()}")

	def collate(batch):
		images = [data["images"] for data in batch]
		images = torch.stack(images)
		images = images.to(memory_format=torch.contiguous_format).float()
		caption = [data["caption"] for data in batch]
		


		return []
		pass

	# pre_dataloader = DataLoader(
	# 	pre_dataset, batch_size=settings["batch_size"], num_workers=8, pin_memory=False,
	# )

	dataset = []
	

	# for i in range(10):
	# 	batch = next(dataloader_iterator)
	# 	print(batch)
	# 	break

if __name__ == "__main__":
	main()