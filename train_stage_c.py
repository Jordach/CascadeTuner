import time
import torch
import torchvision
from torch import nn, optim
from transformers import AutoTokenizer,  CLIPTextModel, CLIPTextModelWithProjection, CLIPVisionModelWithProjection
import transformers

import sys
import os
import math
import copy
import random
import itertools
from core_util import create_folder_if_necessary, load_or_fail, save_model, update_weights_ema, count_total_params, count_trainable_params
from gdf_util import GDF, EpsilonTarget, CosineSchedule, VPScaler, CosineTNoiseCond, DDPMSampler, P2LossWeight, AdaptiveLossWeight
from model_util import EfficientNetEncoder, StageC, StageC_SpectraOne, enable_checkpointing_for_stable_cascade_blocks
from dataset_util import BucketWalker, CachedLatents, RegularLatents
from xformers_util import convert_state_dict_mha_to_normal_attn
from optim_util import step_adafactor, get_optimizer
from bucketeer import Bucketeer
from torch.utils.checkpoint import checkpoint
from diffusers.optimization import get_scheduler

from torchtools.transforms import SmartCrop

from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
from contextlib import contextmanager, nullcontext
from tqdm import tqdm
import yaml
import json
import numpy as np


# Handle special command line args
import argparse
parser = argparse.ArgumentParser(description="Simpler example of a Cascade training script.")
parser.add_argument("--yaml", default=None, type=str, help="The training configuration YAML")
parser.add_argument("--cache_only", default=False, action="store_true", help="Whether to quit after latent caching.")
args = parser.parse_args()

models = {}
settings = {}
info = {
	#"ema_loss": "",
	#"adaptive_loss": {}
}

def load_model(model, model_id=None, full_path=None, strict=True, settings=None, accelerator=None):
	if model_id is not None and full_path is None:
		full_path = f"{settings['checkpoint_path']}/{settings['experiment_id']}/{model_id}.{settings['checkpoint_extension']}"
	elif full_path is None and model_id is None:
		raise ValueError("Loading a model expects full_path or model_id to be defined.")

	ckpt = load_or_fail(full_path, wandb_run_id=None)
	if ckpt is not None:

		if settings["flash_attention"]:
			ckpt = convert_state_dict_mha_to_normal_attn(ckpt)

		model.load_state_dict(ckpt, strict=strict)
		if accelerator.is_main_process:
			print("Loaded requested model from disk.")
		del ckpt
	return model


def text_cache(dropout, text_model, accelerator, captions, att_mask, tokenizer, settings, batch_size):
	text_embeddings = None
	text_embeddings_pool = None

	# Token concatenation things:
	max_length = tokenizer.model_max_length
	max_standard_tokens = max_length - 2
	token_chunks_limit = math.ceil(settings["max_token_limit"] / max_standard_tokens)

	if token_chunks_limit < 1:
		token_chunks_limit = 1

	if dropout:
		# Do not train the text encoder when getting empty embeds
		if settings["train_text_encoder"]:
			text_model.eval()
		captions_unpooled = ["" for _ in range(batch_size)]
		clip_tokens_unpooled = tokenizer(captions_unpooled, truncation=True, padding="max_length",
										max_length=tokenizer.model_max_length,
										return_tensors="pt").to(accelerator.device)

		text_encoder_output = accelerator.unwrap_model(text_model)(**clip_tokens_unpooled, output_hidden_states=True) if accelerator.num_processes > 1 else text_model(**clip_tokens_unpooled, output_hidden_states=True)
		text_embeddings = text_encoder_output.hidden_states[settings["clip_skip"]]
		text_embeddings_pool = text_encoder_output.text_embeds.unsqueeze(1) if "text_embeds" in text_encoder_output else None
		# Restore training mode for the text encoder
		if settings["train_text_encoder"]:
			text_model.train()
	else:
		for chunk_id in range(len(captions)):
			# Hard limit the tokens to fit in memory for the rare event that latent caches that somehow exceed the limit.
			if chunk_id > (token_chunks_limit):
				break

			token_chunk = captions[chunk_id].to(accelerator.device)
			token_chunk = torch.cat((torch.full((token_chunk.shape[0], 1), tokenizer.bos_token_id).to(accelerator.device), token_chunk, torch.full((token_chunk.shape[0], 1), tokenizer.eos_token_id).to(accelerator.device)), 1)
			attn_chunk = att_mask[chunk_id].to(accelerator.device)
			attn_chunk = torch.cat((torch.full((attn_chunk.shape[0], 1), 1).to(accelerator.device), attn_chunk, torch.full((attn_chunk.shape[0], 1), 1).to(accelerator.device)), 1)
			# First 75 tokens we allow BOS to not be masked - otherwise we mask them out
			#if chunk_id == 0:
			#	attn_chunk = torch.cat((torch.full((attn_chunk.shape[0], 1), 1).to(accelerator.device), attn_chunk, torch.full((attn_chunk.shape[0], 1), 0).to(accelerator.device)), 1)
			#else:
			#	attn_chunk = torch.cat((torch.full((attn_chunk.shape[0], 1), 0).to(accelerator.device), attn_chunk, torch.full((attn_chunk.shape[0], 1), 0).to(accelerator.device)), 1)
			text_encoder_output = accelerator.unwrap_model(text_model)(**{"input_ids": token_chunk, "attention_mask": attn_chunk}, output_hidden_states=True) if accelerator.num_processes > 1 else text_model(**{"input_ids": token_chunk, "attention_mask": attn_chunk}, output_hidden_states=True)

			if text_embeddings is None:
				text_embeddings = text_encoder_output["hidden_states"][settings["clip_skip"]]
				text_embeddings_pool = text_encoder_output.text_embeds.unsqueeze(1) if "text_embeds" in text_encoder_output else None
			else:
				text_embeddings = torch.cat((text_embeddings, text_encoder_output["hidden_states"][settings["clip_skip"]]), dim=-2)
				# text_embeddings_pool = torch.cat((text_embeddings_pool, text_encoder_output.text_embeds.unsqueeze(1)), dim=-2) if "text_embeds" in text_encoder_output else None

	return text_embeddings, text_embeddings_pool

# Replaced WarpCore with a more simplified version of it
# made compatible with HF Accelerate
def main():
	global settings
	global info
	global models

	# Basic Setup
	settings["checkpoint_extension"] = "safetensors"
	settings["clip_image_model_name"] = "openai/clip-vit-large-patch14"
	settings["clip_text_model_name"] = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
	settings["clip_tokeniser"] = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
	settings["num_epochs"] = 1
	settings["save_every_n_epoch"] = 1
	settings["clip_skip"] = -1
	settings["max_token_limit"] = 75
	settings["create_latent_cache"] = False
	settings["cache_text_encoder"] = False
	settings["use_latent_cache"] = False
	settings["seed"] = 123
	settings["use_pytorch_cross_attention"] = False
	settings["flash_attention"] = False
	settings["multi_aspect_ratio"] = [1/1, 1/2, 1/3, 2/3, 3/4, 1/5, 2/5, 3/5, 4/5, 1/6, 5/6, 9/16]
	settings["adaptive_loss_weight"] = False
	settings["loss_floor"] = 0
	settings["train_text_encoder"] = False
	settings["lr_scheduler"] = "constant_with_warmup"
	settings["text_lr_scheduler"] = "constant_with_warmup"
	settings["grad_accum_steps"] = 1
	settings["tag_shuffling"] = False
	settings["retokenise_from_cache"] = False
	settings["generator_checkpointing"] = True
	settings["clip_checkpointing"] = True
	settings["generator_optim"] = "_____no_path.pt"
	settings["text_enc_optim"] = "_____no_path.pt"
	settings["skip_steps"] = -1

	gdf = GDF(
		schedule=CosineSchedule(clamp_range=[0.0001, 0.9999]),
		input_scaler=VPScaler(), target=EpsilonTarget(),
		noise_cond=CosineTNoiseCond(),
		loss_weight=AdaptiveLossWeight() if settings["adaptive_loss_weight"] else P2LossWeight(),
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

	if settings["train_text_encoder"] and settings["cache_text_encoder"]:
		raise ValueError("train_text_encoder and cache_text_encoder cannot both be enabled")

	# Copy unet training settings if the supplied config lacks them
	if "text_lr" not in settings:
		settings["text_lr"] = settings["lr"]
	
	if "text_optimizer_type" not in settings:
		settings["text_optimizer_type"] = settings["optimizer_type"]

	if "text_warmup_steps" not in settings:
		settings["text_warmup_steps"] = settings["warmup_updates"]

	main_dtype = getattr(torch, settings["dtype"]) if "dtype" in settings else torch.float32
	if settings["dtype"] == "tf32":
		torch.backends.cuda.matmul.allow_tf32 = True
		torch.backends.cudnn.allow_tf32 = True
	
	accelerator = Accelerator(
		gradient_accumulation_steps=settings["grad_accum_steps"],
		log_with="tensorboard",
		project_dir=f"{settings['checkpoint_path']}"
	)

	# Ensure text encoder and unet paths exist
	unet_path = f"{settings['checkpoint_path']}/unet/"
	tenc_path = f"{settings['checkpoint_path']}/text/"
	if accelerator.is_main_process:
		if not os.path.exists(unet_path):
			os.makedirs(unet_path)
		if not os.path.exists(tenc_path) and settings["train_text_encoder"]:
			os.makedirs(tenc_path)

	if settings["use_pytorch_cross_attention"]:
		if accelerator.is_main_process:
			print("Activating efficient cross attentions.")
		torch.backends.cuda.enable_math_sdp(True)
		torch.backends.cuda.enable_flash_sdp(True)
		torch.backends.cuda.enable_mem_efficient_sdp(True)

	settings["transforms"] = torchvision.transforms.Compose([
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Resize(settings["image_size"], interpolation=torchvision.transforms.InterpolationMode.LANCZOS, antialias=True),
		SmartCrop(settings["image_size"], randomize_p=0.3, randomize_q=0.2)
	])

	full_path = f"{settings['checkpoint_path']}/info.json"
	info_dict = load_or_fail(full_path, wandb_run_id=None) or {}
	info = info | info_dict
	set_seed(settings["seed"])

	# Setup GDF buckets when resuming a training run
	if "adaptive_loss" in info:
		if "bucket_ranges" in info["adaptive_loss"] and "bucket_losses" in info["adaptive_loss"]:
			gdf.loss_weight.bucket_ranges = torch.tensor(info["adaptive_loss"]["bucket_ranges"])
			gdf.loss_weight.bucket_losses = torch.tensor(info["adaptive_loss"]["bucket_losses"])

	# Model Loading For Latent Caching
	# EfficientNet
	if accelerator.is_main_process:
		print("Loading EfficientNetEncoder")
	effnet = EfficientNetEncoder()
	effnet_checkpoint = load_or_fail(settings["effnet_checkpoint_path"])
	effnet.load_state_dict(effnet_checkpoint if "state_dict" not in effnet_checkpoint else effnet_checkpoint["state_dict"])
	effnet.eval().requires_grad_(False).to(accelerator.device, dtype=torch.bfloat16)
	del effnet_checkpoint

	# CLIP Encoders
	if accelerator.is_main_process:
		print("Loading CLIP Text Encoder")
	if settings["model_version"] == "spectraone":
		text_model = CLIPTextModel.from_pretrained(settings["clip_text_model_name"]).to(accelerator.device, dtype=main_dtype if not settings["train_text_encoder"] else torch.float32)
	else:
		text_model = CLIPTextModelWithProjection.from_pretrained(settings["clip_text_model_name"]).to(accelerator.device, dtype=main_dtype if not settings["train_text_encoder"] else torch.float32)
	if accelerator.is_main_process:
		print("Loading CLIP Image Encoder")
	image_model = CLIPVisionModelWithProjection.from_pretrained(settings["clip_image_model_name"]).requires_grad_(False).to(accelerator.device, dtype=main_dtype)
	image_model.eval()

	# Turn on text encoder training if used
	if settings["train_text_encoder"]:
		text_model.requires_grad_(True)
		text_model.train()
		if settings["clip_checkpointing"]:
			text_model.gradient_checkpointing_enable()
	else:
		text_model.requires_grad_(False)
		text_model.eval()

	pre_dataset = []

	tokenizer = AutoTokenizer.from_pretrained(settings["clip_tokeniser"])
	# Setup Dataloader:
	# Only load from the dataloader when not latent caching
	if not settings["use_latent_cache"]:
		print("Loading Dataset[s].")
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

		print("Buckets")

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
		tokens = tokenizer.pad(
			{"input_ids": raw_tokens},
			padding="max_length",
			max_length=len_input,
			return_tensors="pt",
		).to("cpu")
		b_tokens = tokens["input_ids"]
		b_att_mask = tokens["attention_mask"]

		max_standard_tokens = tokenizer.model_max_length - 2
		true_len = max(len(x) for x in b_tokens)
		n_chunks = np.ceil(true_len / max_standard_tokens).astype(int)
		max_len = n_chunks.item() * max_standard_tokens

		# Properly manage memory here - don't bother loading tokens onto GPU.
		# Should prevent an OOM scenario on the GPU.
		cropped_tokens = [b_tokens[:, i:i + max_standard_tokens].clone().detach().to("cpu") for i in range(0, max_len, max_standard_tokens)]
		cropped_attn = [b_att_mask[:, i:i + max_standard_tokens].clone().detach().to("cpu") for i in range(0, max_len, max_standard_tokens)]

		del tokens
		del b_tokens
		del b_att_mask
		
		return {"images": images, "tokens": cropped_tokens, "att_mask": cropped_attn, "caption": caption, "aspects": aspects, "dropout": False}

	pre_dataloader = DataLoader(
		pre_dataset, batch_size=settings["batch_size"], shuffle=False, collate_fn=pre_collate, pin_memory=False,
	)

	# Create second dataset so all images are batched if we're either caching latents or loading direct from disk
	dataset = []

	# Skip initial dataloading pass if we're using a latent cache
	if not settings["use_latent_cache"]:
		for batch in tqdm(pre_dataloader, desc="Dataloader Warmup"):
			dataset.append(batch)

	auto_bucketer = Bucketeer(
		density=settings["image_size"] ** 2,
		factor=32,
		ratios=settings["multi_aspect_ratio"],
		p_random_ratio=0,
		transforms=torchvision.transforms.ToTensor(),
		settings=settings
	)

	# Add duplicate dropout batches with a sufficient amount of steps only when not creating or using a latent cache
	if settings["dropout"] > 0 and not (settings["use_latent_cache"] or settings["create_latent_cache"]):
		dataset_len = len(dataset)
		if dataset_len > 100 and not settings["create_latent_cache"]:
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
		images = images.to(memory_format=torch.contiguous_format)
		images = images.to(accelerator.device)
		tokens = batch[0]["tokens"]
		att_mask = batch[0]["att_mask"]
		captions = batch[0]["caption"]
		return {"images": images, "tokens": tokens, "att_mask": att_mask, "captions": captions, "dropout": False}

	# Shuffle the dataset and initialise the dataloader if we're not latent caching
	set_seed(settings["seed"])
	if not settings["create_latent_cache"]:
		random.shuffle(dataset)
		
	dataloader = DataLoader(
		dataset, batch_size=1, collate_fn=collate, shuffle=False, pin_memory=False
	)

	# Uncomment this to figure out what's wrong with the dataloader:
	# for batch in tqdm(dataloader):
	# 	pass
	# return

	# Optional Latent Caching Step:
	te_dropout, pool_dropout = text_cache(True, text_model, accelerator, [], [], tokenizer, settings, settings["batch_size"])
	def latent_collate(batch):
		cache = torch.load(batch[0]["path"])
		if "dropout" in batch:
			cache[0]["dropout"] = True
		return cache

	latent_cache = CachedLatents(accelerator=accelerator, tokenizer=tokenizer, tag_shuffle=settings["tag_shuffling"], retokenise=settings["retokenise_from_cache"])
	# Create a latent cache if we're not going to load an existing one.
	if settings["create_latent_cache"] and not settings["use_latent_cache"]:
		create_folder_if_necessary(settings["latent_cache_location"])
		step = 0
		for batch in tqdm(dataloader, desc="Latent Caching"):
			with torch.no_grad():
				batch["effnet_cache"] = effnet(effnet_preprocess(batch["images"].to(dtype=main_dtype)))
				batch["clip_cache"] = image_model(clip_preprocess(batch["images"])).image_embeds
				if settings["cache_text_encoder"]:
					te_cache, pool_cache = text_cache(False, text_model, accelerator, batch["tokens"], batch["att_mask"], tokenizer, settings, settings["batch_size"])
					batch["text_cache"] = te_cache
					batch["pool_cache"] = pool_cache
			del batch["images"]
			
			file_name = f"latent_cache_{step}.pt" if not settings["cache_text_encoder"] else f"latent_cache_te_{step}.pt"
			torch.save(batch, os.path.join(settings["latent_cache_location"], file_name))
			latent_cache.add_cache_location(os.path.join(settings["latent_cache_location"], file_name), False)
			step += 1
		if args.cache_only:
			return 0
	
	elif settings["use_latent_cache"]:
		# Load all latent caches from disk. Note that batch size is ignored here and can theoretically be mixed.
		if not os.path.exists(settings["latent_cache_location"]):
			raise Exception("Latent Cache folder does not exist. Please run latent caching first.")

		if len(os.listdir(settings["latent_cache_location"])) == 0:
			raise Exception("No latent caches to load. Please run latent caching first.")
		
		if accelerator.is_main_process:
			print("Loading media from the Latent Cache.")
		for cache in os.listdir(settings["latent_cache_location"]):
			latent_path = os.path.join(settings["latent_cache_location"], cache)
			latent_cache.add_cache_location(latent_path, False)

	# Handle duplicates for Latent Caching
	if settings["create_latent_cache"] or settings["use_latent_cache"]:
		if settings["dropout"] > 0:
			if len(latent_cache) > 100:
				if accelerator.is_main_process:
					print(f"Original Cached Step Count: {len(latent_cache)}")
				total_batches = int((len(latent_cache)-1) * settings["dropout"])

				dropouts = random.sample(latent_cache.get_cache_list(), total_batches)
				for batch in dropouts:
					latent_cache.add_cache_location(batch[0], True)

				if accelerator.is_main_process:
					print(f"Duplicated {len(dropouts)} caches for caption dropout.")
					print(f"Total Cached Step Count: {len(latent_cache)}")
		
		dataloader = torch.utils.data.DataLoader(
			latent_cache, batch_size=1, collate_fn=lambda x: x, shuffle=False, 
		)

	# Special things
	@contextmanager
	def loading_context():
		yield None

	# Load in Stage C/B
	with loading_context():
		if "model_version" not in settings:
			raise ValueError('model_version key is missing from supplied YAML.')
		
		flash_attention = settings["flash_attention"]
		# generator_ema = None
		if settings["model_version"] == "3.6B":
			if accelerator.is_main_process:
				print("Creating and loading an instance of Stage C 3.6B.")
			generator = StageC(flash_attention=flash_attention)
			# if "ema_start_iters" in settings:
				# generator_ema = StageC(flash_attention=flash_attention)
		elif settings["model_version"] == "1B":
			if accelerator.is_main_process:
				print("Creating and loading an instance of Stage C 1B.")
			generator = StageC(c_cond=1536, c_hidden=[1536, 1536], nhead=[24, 24], blocks=[[4, 12], [12, 4]], flash_attention=flash_attention)
			# if "ema_start_iters" in settings:
				# generator_ema = StageC(c_cond=1536, c_hidden=[1536, 1536], nhead=[24, 24], blocks=[[4, 12], [12, 4]], flash_attention=flash_attention)
		elif settings["model_version"] == "spectraone":
			if accelerator.is_main_process:
				print("Creating and loading an instance of Spectra One Stage C 1B.")
			generator = StageC_SpectraOne(c_cond=1536, c_hidden=[1536, 1536], blocks=[[6, 12], [12, 6]], nhead=[64, 64], c_clip_text=768, c_clip_text_pooled=768, flash_attention=flash_attention)
			print()
		else:
			raise ValueError(f"Unknown model size: {settings['model_version']}, stopping.")

	if accelerator.is_main_process:
		print(f"Model Total Params:     {count_total_params(generator)}")
		print(f"Model Trainable Params: {count_trainable_params(generator)}")

	if "generator_checkpoint_path" in settings:
		# generator.load_state_dict(load_or_fail(settings["generator_checkpoint_path"]))
		generator = load_model(generator, model_id=None, full_path=settings["generator_checkpoint_path"], settings=settings, accelerator=accelerator)
		# import optree
		# optree.tree_map(lambda x: print(x.dtype), generator.state_dict())
		# return
	else:
		if accelerator.is_main_process:
			print("WARNING: Initialising model from empty weights. Please ensure your YAML contains generator_checkpoint_path.")
		# generator = load_model(generator, model_id='generator', settings=settings, accelerator=accelerator)
	
	if settings["generator_checkpointing"]:
		enable_checkpointing_for_stable_cascade_blocks(generator, accelerator.device)
	generator = generator.to(accelerator.device, dtype=main_dtype)

	# if generator_ema is not None:
	# 	generator_ema.load_state_dict(generator.state_dict())
	# 	generator_ema = load_model(generator_ema, "generator_ema", settings=settings, accelerator=accelerator)
	# 	generator_ema.to(accelerator.device, dtype=main_dtype)

	# Load optimizers and LR schedules
	if accelerator.is_main_process:
		print("Loading optimizer[s].")

	# Divide up the dataset cache onto the other GPUs
	accelerator.wait_for_everyone()
	dataloader = accelerator.prepare(dataloader)

	# Unet/Stage C optimizer and schedule
	unet_optimizer, unet_optimizer_kwargs = get_optimizer(settings["optimizer_type"], settings)

	unet_params = (
		generator.parameters()
	)
	unet_optimizer = unet_optimizer(unet_params, lr=settings["lr"], **unet_optimizer_kwargs)
	if os.path.exists(settings["generator_optim"]):
		unet_optimizer.load_state_dict(torch.load(settings["generator_optim"]))
		# Reset the LR to the new values
		for param in unet_optimizer.param_groups:
			param["lr"] = settings["lr"]
	elif settings["generator_optim"] == "_____no_path.pt":
		pass
	else:
		raise ValueError("Cannot load Stage C optimizer state from disk, does it exist?")

	# Special hook for stochastic rounding for adafactor
	if settings["optimizer_type"].lower() == "adafactorstoch":
		unet_optimizer.step = step_adafactor.__get__(unet_optimizer, transformers.optimization.Adafactor)

	unet_scheduler = get_scheduler(
		settings["lr_scheduler"],
		optimizer=unet_optimizer,
		num_warmup_steps=settings["warmup_updates"],
		num_training_steps=len(dataloader) * settings["num_epochs"]
	)

	# Text Encoder optimizer and LR schedule
	# Make dummy variables to keep them always available
	text_optimizer = ""
	text_params = ""
	text_scheduler = ""
	if settings["train_text_encoder"]:
		text_optimizer, text_optimizer_kwargs = get_optimizer(settings["text_optimizer_type"], settings)

		text_params = (
			text_model.parameters()
		)

		text_optimizer = text_optimizer(text_params, lr=settings["text_lr"], **text_optimizer_kwargs)
		if os.path.exists(settings["text_enc_optim"]) and settings["text_enc_optim"] != "_____no_path.pt":
			text_optimizer.load_state_dict(torch.load(settings["text_enc_optim"]))
			# Reset the LR to the new values
			for param in text_optimizer.param_groups:
				param["lr"] = settings["text_lr"]
		elif settings["text_enc_optim"] == "_____no_path.pt":
			pass
		else:
			raise ValueError("Cannot load Text Encoder optimizer state from disk, does it exist?")

		text_scheduler = get_scheduler(
			settings["text_lr_scheduler"],
			optimizer=text_optimizer,
			num_warmup_steps=settings["warmup_updates"],
			num_training_steps=len(dataloader) * settings["num_epochs"]
		)

	# Prepare everything at the same time
	generator, text_model = accelerator.prepare(generator, text_model)
	unet_optimizer, unet_scheduler = accelerator.prepare(unet_optimizer, unet_scheduler)
	if settings["train_text_encoder"]:
		text_optimizer, text_scheduler = accelerator.prepare(text_optimizer, text_scheduler)

	if accelerator.is_main_process:
		accelerator.init_trackers("training")

	# Training loop
	steps_bar = tqdm(range(len(dataloader)), desc="Steps to Epoch", disable=not accelerator.is_local_main_process)
	epoch_bar = tqdm(range(settings["num_epochs"]), desc="Epochs", disable=not accelerator.is_local_main_process)
	total_steps = 0

	# Special case for handling latent caching
	# saves one second of time to avoid expensive key checking
	# This is enabled if we've just finished latent caching and want to immediately start training thereafter
	is_latent_cache = False
	if settings["use_latent_cache"] or settings["create_latent_cache"]:
		is_latent_cache = True
		del image_model
		if settings["cache_text_encoder"]:
			del text_model
		del effnet
		torch.cuda.empty_cache()

	# Handle 
	text_encoder_context = nullcontext() if settings["train_text_encoder"] else torch.no_grad()
	last_grad_norm = 0
	generator.train()
	for e in epoch_bar:
		current_step = 0
		steps_bar.reset(total=len(dataloader))
		for step, batch in enumerate(dataloader):
			if total_steps < settings["skip_steps"] + 1 and settings["skip_steps"] > 0:
				steps_bar.update(1)
				current_step += 1
				total_steps += 1
				accelerator.wait_for_everyone()
				continue

			with accelerator.accumulate(generator, text_model) if settings["train_text_encoder"] else accelerator.accumulate(generator):
				captions = batch[0]["tokens"]
				attn_mask = batch[0]["att_mask"]
				images = batch[0]["images"] if not is_latent_cache else None
				dropout = batch[0]["dropout"]
				batch_size = len(batch[0]["captions"])

				with text_encoder_context:
					text_embeddings = None
					text_embeddings_pool = None
					if is_latent_cache:
						if "text_cache" in batch[0] and "pool_cache" in batch[0]:
							text_embeddings = batch[0]["text_cache"]
							text_embeddings_pool = batch[0]["pool_cache"]
						else:
							text_embeddings, text_embeddings_pool = text_cache(dropout, text_model, accelerator, captions, attn_mask, tokenizer, settings, batch_size)
					else:
						text_embeddings, text_embeddings_pool = text_cache(dropout, text_model, accelerator, captions, attn_mask, tokenizer, settings, batch_size)
					
				with torch.no_grad():
					# Handle Image Encoding
					image_embeddings = torch.zeros(batch_size, 768, device=accelerator.device, dtype=main_dtype)
					if not dropout:
						rand_id = np.random.rand(batch_size) > 0.9
						if any(rand_id):
							image_embeddings[rand_id] = image_model(clip_preprocess(images[rand_id].to(dtype=main_dtype))).image_embeds.to(dtype=main_dtype) if not is_latent_cache else batch[0]["clip_cache"][rand_id].to(dtype=main_dtype)
					image_embeddings = image_embeddings.unsqueeze(1)

					# Get Latents
					latents = effnet(effnet_preprocess(images.to(dtype=main_dtype))) if not is_latent_cache else batch[0]["effnet_cache"]
					noised, noise, target, logSNR, noise_cond, loss_weight = gdf.diffuse(latents.to(dtype=main_dtype), shift=1, loss_shift=1)

				# Forwards Pass
				with accelerator.autocast():
					pred = generator(noised, noise_cond, 
						**{
							"clip_text": text_embeddings.to(dtype=main_dtype),
							"clip_text_pooled": text_embeddings_pool.to(dtype=main_dtype) if text_embeddings_pool is not None else text_embeddings_pool,
							"clip_img": image_embeddings.to(dtype=main_dtype)
						}
					)
					loss = nn.functional.mse_loss(pred, target, reduction="none").mean(dim=[1,2,3])
					loss_adjusted = (loss * loss_weight).mean()
				# And convert to fp32 
				loss_adjusted = loss_adjusted.to(dtype=torch.float32, device=accelerator.device)

				if isinstance(gdf.loss_weight, AdaptiveLossWeight):
					gdf.loss_weight.update_buckets(logSNR, loss)

				# Backwards Pass
				accelerator.backward(loss_adjusted)

				if accelerator.sync_gradients:
					grad_norm = accelerator.clip_grad_norm_(itertools.chain(generator.parameters(), text_model.parameters()) if settings["train_text_encoder"] else generator.parameters(), 1.0)
					last_grad_norm = grad_norm.mean().item()
				
				unet_optimizer.step()
				unet_scheduler.step()
				unet_optimizer.zero_grad()

				if settings["train_text_encoder"]:
					text_optimizer.step()
					text_scheduler.step()
					text_optimizer.zero_grad()

				steps_bar.update(1)
				current_step += 1
				total_steps += 1

				# Handle EMA weights
				# if generator_ema is not None and current_step % settings["ema_iters"] == 0:
				# 	update_weights_ema(
				# 		generator_ema, generator,
				# 		beta=(settings["ema_beta"] if current_step > settings["ema_start_iters"] else 0)
				# 	)

			if accelerator.is_main_process:
				logs = {
					"loss": loss_adjusted.item(),
					"grad_norm": last_grad_norm,
					"lr": unet_scheduler.get_last_lr()[0]
				}

				if settings["train_text_encoder"]:
					logs["te_lr"] = text_scheduler.get_last_lr()[0]

				epoch_bar.set_postfix(logs)
				accelerator.log(logs, step=total_steps)

			if (current_step) % settings["save_every"] == 0:
				if accelerator.is_main_process:
					save_model(
						accelerator.unwrap_model(generator), 
						model_id = f"unet/{settings['experiment_id']}", settings=settings, accelerator=accelerator, step=f"e{e}_s{current_step}"
					)
					torch.save(
						accelerator.unwrap_model(unet_optimizer).state_dict(),
						os.path.join(unet_path, f"stage_c_optimizer_e{e}_s{current_step}.pth")
					)

					if settings["train_text_encoder"]:
						if accelerator.num_processes > 1:
							accelerator.unwrap_model(text_model).save_pretrained(os.path.join(tenc_path, f"{settings['experiment_id']}_e{e}_s{current_step}_te/"))
						else:
							text_model.save_pretrained(os.path.join(tenc_path, f"{settings['experiment_id']}_e{e}_s{current_step}_te/"))
						tokenizer.save_vocabulary(os.path.join(tenc_path, f"{settings['experiment_id']}_e{e}_s{current_step}_te/"))
						torch.save(
							accelerator.unwrap_model(text_optimizer).state_dict(), 
							os.path.join(tenc_path, f"te_optimizer_e{e}_s{current_step}.pth")
						)
				accelerator.wait_for_everyone()

		if (e+1) % settings["save_every_n_epoch"] == 0 or settings["save_every_n_epoch"] == 1:
			if accelerator.is_main_process:
				save_model(
					accelerator.unwrap_model(generator), 
					model_id = f"unet/{settings['experiment_id']}", settings=settings, accelerator=accelerator, step=f"e{e+1}"
				)
				torch.save(
					accelerator.unwrap_model(unet_optimizer).state_dict(),
					os.path.join(unet_path, f"stage_c_optimizer_e{e+1}.pth")
				)

				if settings["train_text_encoder"]:
					if accelerator.num_processes > 1:
						accelerator.unwrap_model(text_model).save_pretrained(os.path.join(tenc_path, f"{settings['experiment_id']}_e{e+1}_te/"))
					else:
						text_model.save_pretrained(os.path.join(tenc_path, f"{settings['experiment_id']}_e{e+1}_te/"))
					tokenizer.save_vocabulary(os.path.join(tenc_path, f"{settings['experiment_id']}_e{e+1}_te/"))
					torch.save(
						accelerator.unwrap_model(text_optimizer).state_dict(), 
						os.path.join(tenc_path, f"te_optimizer_e{e+1}.pth")
					)
			accelerator.wait_for_everyone()
		
		settings["seed"] += 1
		set_seed(settings["seed"])

if __name__ == "__main__":
	main()