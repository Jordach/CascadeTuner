import torch
import random
import os
from torch.utils.data import Dataset
from tokeniser_util import tokenize_respecting_boundaries
from zstd_util import load_torch_zstd
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

def vae_encode(images, vae):
	_images = images.to(dtype=vae.dtype)
	latents = vae.encode(_images).latent_dist.sample()
	return latents * 0.18215

# This is known to work on multi-GPU setups
class SD1CachedLatents(Dataset):
	def __init__(self, accelerator, tokenizer=None, tag_shuffle=True):
		self.batches = []
		self.accelerator = accelerator
		self.tokenizer = tokenizer
		self.tag_shuffle = tag_shuffle
		if tag_shuffle:
			self.accelerator.print("Will shuffle captions in Latent Caches.")

	def __len__(self):
		return len(self.cache_paths)

	def __getitem__(self, index):
		if index == 0:
			random.shuffle(self.cache_paths)
			self.accelerator.print("Cached Latents Shuffled.")

		exts = os.path.splitext(self.cache_paths[index][0])
		if exts[1] == ".pt":
			cache = torch.load(self.cache_paths[index][0], map_location=self.accelerator.device)
		elif exts[1] == ".zpt":
			cache = load_torch_zstd(self.cache_paths[index][0], self.accelerator.device)
		else:
			raise ValueError(f"Unknown Latent Cache format for file: {self.cache_paths[index][0]}")
		
		if self.cache_paths[index][1]:
			cache["dropout"] = True

		if self.tag_shuffle:
			del cache["tokens"]
			del cache["att_mask"]

			shuffled_captions = []
			for caption in cache["captions"]:
				tags = caption.split(",")
				random.shuffle(tags)
				shuffled_caption = ", ".join(tag.strip() for tag in tags)
				shuffled_captions.append(shuffled_caption)

			# Tokenize with our custom function that respects word boundaries
			tokenized_captions, attention_masks = tokenize_respecting_boundaries(self.tokenizer, shuffled_captions)

			cache["tokens"] = tokenized_captions
			cache["att_mask"] = attention_masks

		return cache

	def get_batch_list(self):
		return self.batches
	
	def add_latent_batch(self, batch, dropout):
		self.batches.append((batch, dropout))

def save_sd1_pipeline(path, settings, accelerator, unet, text_model):
	if accelerator.is_main_process:
		pipeline = StableDiffusionPipeline.from_pretrained(settings["model_name"])
		pipeline.unet = accelerator.unwrap_model(unet)
		if settings["train_text_encoder"]:
			pipeline.text_encoder = accelerator.unwrap_model(text_model)
		pipeline.save_pretrained(path)