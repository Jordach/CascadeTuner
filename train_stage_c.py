import torch
import torchvision
from torch import nn, optim
from transformers import AutoTokenizer, CLIPTextModelWithProjection, CLIPVisionModelWithProjection
import transformers
#from diffusers.optimization import get_scheduler

import sys
import os
import math
import copy
import random
from core_util import create_folder_if_necessary, load_or_fail, load_optimizer, save_model, save_optimizer, update_weights_ema
from gdf_util import GDF, EpsilonTarget, CosineSchedule, VPScaler, CosineTNoiseCond, DDPMSampler, P2LossWeight, AdaptiveLossWeight
from model_util import EfficientNetEncoder, StageC, ResBlock, AttnBlock, TimestepBlock, FeedForwardBlock
from dataset_util import BucketWalker
from bucketeer import Bucketeer
from warmup_scheduler import GradualWarmupScheduler
from fractions import Fraction

from torchtools.transforms import SmartCrop

from torch.utils.data import DataLoader
from accelerate import init_empty_weights, Accelerator
from accelerate.utils import set_module_tensor_to_device, set_seed
from contextlib import contextmanager
from tqdm import tqdm
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
	settings["wandb_project"] = None
	settings["wandb_entity"] = None
	settings["clip_image_model_name"] = "openai/clip-vit-large-patch14"
	settings["clip_text_model_name"] = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
	settings["num_epochs"] = 1
	settings["save_every_n_epoch"] = 1
	settings["clip_skip"] = -1
	settings["max_token_limit"] = 75
	
	gdf_loss_weight = P2LossWeight()
	if "adaptive_loss_weight" in settings:
		if settings["adaptive_loss_weight"]:
			gdf_loss_weight = AdaptiveLossWeight()

	gdf = GDF(
		schedule=CosineSchedule(clamp_range=[0.0001, 0.9999]),
		input_scaler=VPScaler(), target=EpsilonTarget(),
		noise_cond=CosineTNoiseCond(),
		loss_weight=gdf_loss_weight,
	)

	effnet_preprocess = torchvision.transforms.Compose([
		torchvision.transforms.Normalize(
			mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
		)
	])

	clip_preprocess = torchvision.transforms.Compose([
		torchvision.transforms.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
		torchvision.transforms.CenterCrop(224),
		torchvision.transforms.Normalize(
			mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
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

	if "use_pytorch_cross_attention" in settings:
		if settings["use_pytorch_cross_attention"]:
			print("Activating efficient cross attentions.")
			torch.backends.cuda.enable_math_sdp(True)
			torch.backends.cuda.enable_flash_sdp(True)
			torch.backends.cuda.enable_mem_efficient_sdp(True)

	settings["transforms"] = torchvision.transforms.Compose([
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Resize(settings["image_size"], interpolation=torchvision.transforms.InterpolationMode.LANCZOS, antialias=True),
		SmartCrop(settings["image_size"], randomize_p=0.3, randomize_q=0.2)
	])

	full_path = f"{settings['checkpoint_path']}/{settings['experiment_id']}/info.json"
	info_dict = load_or_fail(full_path, wandb_run_id=None) or {}
	info = info | info_dict
	set_seed(settings["seed"] if "seed" in settings else 123)

	# Setup GDF buckets when resuming a training run
	if "adaptive_loss" in info:
		if "bucket_ranges" in info["adaptive_loss"] and "bucket_losses" in info["adaptive_loss"]:
			gdf.loss_weight.bucket_ranges = torch.tensor(info["adaptive_loss"]["bucket_ranges"])
			gdf.loss_weight.bucket_losses = torch.tensor(info["adaptive_loss"]["bucket_losses"])

	hf_accel_dtype = settings["dtype"]
	if settings["dtype"] == "bfloat16":
		hf_accel_dtype = "bf16"
	elif settings["dtype"] == "tf32":
		hf_accel_dtype = "no"
		torch.backends.cuda.matmul.allow_tf32 = True
		torch.backends.cudnn.allow_tf32 = True
	elif settings["dtype"] != "fp16" or settings["dtype"] != "fp32":
		hf_accel_dtype = "no"
	
	accelerator = Accelerator(
		gradient_accumulation_steps=settings["grad_accum_steps"],
		mixed_precision=hf_accel_dtype,
		log_with="tensorboard",
		project_dir=f"{settings['output_path']}"
	)

	# Setup Dataloader:
	print("Loading Dataset[s].")
	tokenizer = AutoTokenizer.from_pretrained(settings["clip_text_model_name"])
	pre_dataset = BucketWalker(
		reject_aspects=settings["reject_aspects"],
		tokenizer=tokenizer
	)

	if "local_dataset_path" in settings:
		if type(settings["local_dataset_path"]) is list:
			for dir in settings["local_dataset_path"]:
				pre_dataset.scan_folder(dir)
		elif type(settings["local_dataset_path"]) is str:
			pre_dataset.scan_folder(settings["local_dataset_path"])
		else:
			raise ValueError("'local_dataset_path' must either be a string, or list of strings containing paths.")

	pre_dataset.bucketize(settings["batch_size"])
	print(f"Total Invalid Files:  {pre_dataset.get_rejects()}")
	settings["multi_aspect_ratio"] = pre_dataset.get_buckets()

	def pre_collate(batch):
		# Do NOT load images - save that for the second dataloader pass
		images = [data["images"] for data in batch]
		caption = [data["caption"] for data in batch]
		raw_tokens = [data["tokens"] for data in batch]
		aspects = [data["aspects"] for data in batch]
		
		# Get total number of chunks
		max_len = max(len(x) for x in raw_tokens)
		num_chunks = math.ceil(max_len / (tokenizer.model_max_length - 2))
		if num_chunks < 1:
			num_chunks = 1
		
		# Get the true padded length of the tokens
		len_input = tokenizer.model_max_length - 2
		if num_chunks > 1:
			len_input = (tokenizer.model_max_length * num_chunks) - (num_chunks * 2)
		
		# Tokenize!
		batch_tokens = tokenizer.pad(
			{"input_ids": raw_tokens},
			padding="max_length",
			max_length=len_input,
			return_tensors="pt"
		).input_ids

		return {"images": images, "tokens": batch_tokens, "caption": caption, "aspects": aspects, "dropout": False}

	pre_dataloader = DataLoader(
	 	pre_dataset, batch_size=settings["batch_size"], shuffle=False, collate_fn=pre_collate, pin_memory=False,
	)

	# Create second dataset so all images are batched
	dataset = []
	for batch in pre_dataloader:
		dataset.append(batch)

	auto_bucketer = Bucketeer(
		density=settings["image_size"] ** 2,
		factor=32,
		ratios=settings["multi_aspect_ratio"],
		p_random_ratio=settings["bucketeer_random_ratio"] if "bucketeer_random_ratio" in settings else 0,
		transforms=torchvision.transforms.ToTensor(),
	)

	# Add duplicate dropout batches with a sufficient amount of steps
	if "dropout" in settings and settings["dropout"] > 0:
		dataset_len = len(dataset)
		if dataset_len > 100:
			dropouts = random.sample(dataset, int(dataset_len * settings["dropout"]))
			new_dropouts = copy.deepcopy(dropouts)
			for batch in new_dropouts:
				batch["dropout"] = True
			dataset.extend(new_dropouts)
			print(f"Duplicated {len(dropouts)} batches for caption dropout.")
			print(f"Updated Step Count: {len(dataset)}")
		else:
			print("Could not create duplicate batches for caption dropout due to insufficient batch counts.")

	def collate(batch):
		images = []
		# The reason for not unrolling the images in the prior dataloader was so we can load them only when training,
		# rather than storing all transformed images in memory!
		aspects = batch[0]["aspects"]
		img = batch[0]["images"]
		for i in range(0, len(batch[0]["images"])):
			images.append(auto_bucketer.load_and_resize(img[i], float(aspects[i])))
		images = torch.stack(images)
		images = images.to(memory_format=torch.contiguous_format).float()
		images = images.to(accelerator.device)
		tokens = batch[0]["tokens"]
		captions = batch[0]["caption"]
		return {"images": images, "tokens": tokens, "captions": captions, "dropout": False}

	# Shuffle the dataset
	set_seed(settings["seed"])
	random.shuffle(dataset)
	dataloader = DataLoader(
		dataset, batch_size=1, collate_fn=collate, shuffle=False, pin_memory=False
	)

	# Setup Models:
	main_dtype = getattr(torch, settings["dtype"]) if settings["dtype"] else torch.float32

	# EfficientNet
	effnet = EfficientNetEncoder()
	effnet_checkpoint = load_or_fail(settings["effnet_checkpoint_path"])
	effnet.load_state_dict(effnet_checkpoint if "state_dict" not in effnet_checkpoint else effnet_checkpoint["state_dict"])
	effnet.eval().requires_grad_(False).to(accelerator.device)
	del effnet_checkpoint

	# Previewer (Not used?)
	# previewer = Previewer()
	# previewer_checkpoint = load_or_fail(settings["previewer_checkpoint_path"])
	# previewer.load_state_dict(previewer_checkpoint if "state_dict" not in previewer_checkpoint else previewer_checkpoint["state_dict"])
	# previewer.eval().requires_grad_(False).to(accelerator.device)
	# del previewer_checkpoint

	# Special things
	@contextmanager
	def loading_context():
		yield None

	# Load in Stage C/B	
	with loading_context():
		if "model_version" not in settings:
			raise ValueError('model_version key is missing from supplied YAML.')
		
		generator_ema = None
		if settings["model_version"] == "3.6B":
			generator = StageC()
			if "ema_start_iters" in settings:
				generator_ema = StageC()
		elif settings["model_version"] == "1B":
			generator = StageC(c_cond=1536, c_hidden=[1536, 1536], nhead=[24, 24], blocks=[[4, 12], [12, 4]])
			if "ema_start_iters" in settings:
				generator_ema = StageC(c_cond=1536, c_hidden=[1536, 1536], nhead=[24, 24], blocks=[[4, 12], [12, 4]])
		else:
			raise ValueError(f"Unknown model size: {settings['model_version']}, stopping.")

	if "generator_checkpoint_path" in settings:
		# generator.load_state_dict(load_or_fail(settings["generator_checkpoint_path"]))
		generator = load_model(generator, model_id=None, full_path=settings["generator_checkpoint_path"])
	else:
		generator = load_model(generator, model_id='generator')
	generator = generator.to(accelerator.device, dtype=main_dtype)

	if generator_ema is not None:
		generator_ema.load_state_dict(generator.state_dict())
		generator_ema = load_model(generator_ema, "generator_ema")
		generator_ema.to(accelerator.device, dtype=main_dtype)
	
	# CLIP Encoders
	text_model = CLIPTextModelWithProjection.from_pretrained(settings["clip_text_model_name"]).requires_grad_(False).to(accelerator.device, dtype=main_dtype)
	text_model.eval()
	image_model = CLIPVisionModelWithProjection.from_pretrained(settings["clip_image_model_name"]).requires_grad_(False).to(accelerator.device, dtype=main_dtype)
	image_model.eval()

	# if accelerator.is_main_process:
	# 	print(yaml.dump(settings, default_flow_style=False))
	# 	print()
	# 	print(yaml.dump(info, default_flow_style=False))
	# 	print()

	# Load optimizers
	optimizer_type = settings["optimizer_type"].lower()
	optimizer_kwargs = {}
	if optimizer_type == "adamw":
		optimizer = optim.AdamW
	elif optimizer_type == "adamw8bit":
		try:
			import bitsandbytes as bnb
		except ImportError:
			raise ImportError("Please ensure bitsandbytes is installed: pip install bitsandbytes")
		optimizer = bnb.optim.AdamW8bit
	else: #AdaFactor
		optimizer_kwargs["scale_parameter"] = False
		optimizer_kwargs["relative_step"] = False
		optimizer_kwargs["warmup_init"] = False
		optimizer = transformers.optimization.Adafactor

	optimizer = optimizer(generator.parameters(), lr=settings["lr"], **optimizer_kwargs)
	optimizer = load_optimizer(optimizer, 'generator_optim', settings=settings)

	# Load scheduler
	scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=settings["warmup_updates"])
	scheduler.last_epoch = info["total_steps"] if "total_steps" in info else None

	accelerator.prepare(generator, dataloader, text_model, image_model, optimizer, scheduler)

	if accelerator.is_main_process:
		accelerator.init_trackers("training")

	# Token concatenation things:
	max_length = tokenizer.model_max_length
	max_standard_tokens = max_length - 2
	token_chunks_limit = math.ceil(settings["max_token_limit"] / max_standard_tokens)
	if token_chunks_limit < 1:
		token_chunks_limit = 1

	# Training loop
	epoch_bar = tqdm(range(settings["num_epochs"]), desc="Epoch")
	steps_bar = tqdm(dataloader, desc="Steps to Epoch")
	step_count = 1

	generator.train()
	total_steps = 0
	for e in epoch_bar:
		current_step = 0
		for batch in steps_bar:
			with accelerator.accumulate(generator):
				# Forwards Pass
				captions = batch["tokens"]
				images = batch["images"]
				dropout = batch["dropout"]
				batch_size = settings["batch_size"]

				# Handle Text Encoding
				text_embeddings = None
				text_embeddings_pool = None

				with torch.no_grad():
					if dropout:
						captions_unpooled = ["" for _ in range(batch_size)]
						clip_tokens_unpooled = tokenizer(captions_unpooled, truncation=True, padding="max_length",
														max_length=models.tokenizer.model_max_length,
														return_tensors="pt").to(accelerator.device)
						text_encoder_output = text_model(**clip_tokens_unpooled, output_hidden_states=True)
						text_embeddings = text_encoder_output.hidden_states[settings["clip_skip"]]
						text_embeddings_pool = text_encoder_output.text_embeds.unsqueeze(1)
					else:
						true_len = max(len(x) for x in captions)
						n_chunks = np.ceil(true_len / max_standard_tokens).astype(int)
						max_len = n_chunks.item() * max_standard_tokens

						token_copy = captions.detach().clone()
						for i, x in enumerate(token_copy):
							if len(x) < max_len:
								token_copy[i] = [*x, *np.full((max_len - len(x)), tokenizer.eos_token_id)]
							del i, x
						
						chunks = [token_copy[:, i:i + max_standard_tokens] for i in range(0, max_len, max_standard_tokens)]
						n_processed = 0
						for chunk in chunks:
							# Hard limit the tokens to fit in memory for the rare event that latent caches that somehow exceed the limit.
							if n_processed > (token_chunks_limit):
								del chunk
								break

							chunk = chunk.to(accelerator.device)
							chunk = torch.cat((torch.full((chunk.shape[0], 1), tokenizer.bos_token_id).to(accelerator.device), chunk, torch.full((chunk.shape[0], 1), tokenizer.eos_token_id).to(accelerator.device)), 1)
							text_encoder_output = text_model(chunk, output_hidden_states=True)
							if text_embeddings is None:
								text_embeddings = text_encoder_output["hidden_states"][settings["clip_skip"]]
								text_embeddings_pool = text_encoder_output.text_embeds.unsqueeze(1)
							else:
								text_embeddings = torch.cat((text_embeddings, text_encoder_output["hidden_states"][settings["clip_skip"]]), dim=-2)
								text_embeddings_pool = torch.cat((text_embeddings_pool, text_encoder_output.text_embeds.unsqueeze(1)), dim=-2)
							n_processed += 1
						del chunks, token_copy
					
				# Handle Image Encoding
				image_embeddings = torch.zeros(batch_size, 768, device=accelerator.device, dtype=torch.float32)
				with torch.no_grad():
					if not dropout:
						rand_id = np.random.rand(batch_size) > 0.9
						if any(rand_id):
							image_embeddings[rand_id] = image_model(clip_preprocess(images[rand_id])).image_embeds
					image_embeddings = image_embeddings.unsqueeze(1)

				# Get Latents
				latents = effnet(effnet_preprocess(images))
				noised, noise, target, logSNR, noise_cond, loss_weight = gdf.diffuse(latents, shift=1, loss_shift=1)
				
				# Forwards Pass
				pred = None
				loss = None
				loss_adjusted = None
				with torch.cuda.amp.autocast(dtype=torch.bfloat16):
					pred = generator(noised, noise_cond, **{"clip_text": text_embeddings, "clip_text_pooled": text_embeddings_pool, "clip_img": image_embeddings})
					loss = nn.functional.mse_loss(pred, target, reduction="none").mean(dim=[1,2,3])
					loss_adjusted = (loss * loss_weight).mean() / settings["grad_accum_steps"]

				if isinstance(gdf.loss_weight, AdaptiveLossWeight):
					gdf.loss_weight.update_buckets(logSNR, loss)

				# Backwards Pass
				accelerator.backward(loss_adjusted)
				grad_norm = nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
				optimizer.step()
				optimizer.zero_grad()

				current_step += 1
				total_steps += 1

				# Handle EMA weights
				if generator_ema is not None and current_step % settings["ema_iters"] == 0:
					update_weights_ema(
						generator_ema, generator,
						beta=(settings["ema_beta"] if current_step > settings["ema_start_iters"] else 0)
					)
				
				info["ema_loss"] = loss.mean().item() if "ema_loss" not in info else info["ema_loss"] * 0.99 + loss.mean().item() * 0.01

				if accelerator.is_main_process:
					logs = {
						"loss": loss_adjusted.mean().item(),
						"ema_loss": info["ema_loss"],
						"grad_norm": grad_norm.item(),
						"lr": settings["lr"]
					}

					epoch_bar.set_postfix(logs)
					accelerator.log(logs, step=total_steps)
		
		if not e % settings["save_every_n_epochs"]:
			if accelerator.is_main_process:
				accelerator.wait_for_everyone()
				save_model(generator if generator_ema is None else generator_ema, model_id = f"generator", settings=settings, accelerator=accelerator)

if __name__ == "__main__":
	main()