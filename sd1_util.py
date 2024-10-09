import torch
import random
import os
from torch.utils.data import Dataset
from tokeniser_util import tokenize_respecting_boundaries, shuffle_and_drop_tags
from zstd_util import load_torch_zstd
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision import transforms
from tqdm import tqdm
import sys
from contextlib import contextmanager
from io import StringIO

def vae_encode(images, vae):
	_images = images.to(dtype=vae.dtype)
	latents = vae.encode(_images).latent_dist.sample() * vae.config.scaling_factor
	return latents

vae_preprocess = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# This is known to work on multi-GPU setups
class SD1CachedLatents(Dataset):
	def __init__(self, accelerator, settings, tokenizer=None, tag_shuffle=True):
		self.batches = []
		self.accelerator = accelerator
		self.settings = settings
		self.tokenizer = tokenizer
		self.tag_shuffle = tag_shuffle
		if tag_shuffle:
			self.accelerator.print("Will shuffle captions in Latent Caches.")

	def __len__(self):
		return len(self.batches)

	def __getitem__(self, index):
		if index == 0:
			random.shuffle(self.batches)
			# if self.accelerator.is_main_process:
			# 	tqdm.write("Latent Cache shuffled.")

		exts = os.path.splitext(self.batches[index][0])
		if exts[1] == ".pt":
			cache = torch.load(self.batches[index][0], map_location=self.accelerator.device)
		elif exts[1] == ".zpt":
			cache = load_torch_zstd(self.batches[index][0], self.accelerator.device)
		else:
			raise ValueError(f"Unknown Latent Cache format for file: {self.batches[index][0]}")
		
		if self.batches[index][1]:
			cache["dropout"] = True

		if self.tag_shuffle:
			del cache["tokens"]
			del cache["att_mask"]

			shuffled_captions = []
			for caption in cache["captions"]:
				shuffled_caption = shuffle_and_drop_tags(caption, self.settings)
				shuffled_captions.append(shuffled_caption.strip())

			# Tokenize with our custom function that respects word boundaries
			tokenized_captions, attention_masks = tokenize_respecting_boundaries(self.tokenizer, shuffled_captions)

			# Really ensure the shuffled_captions are used
			del cache["captions"]
			cache["captions"] = shuffled_captions
			cache["tokens"] = tokenized_captions
			cache["att_mask"] = attention_masks

		return cache

	def get_batch_list(self):
		return self.batches
	
	def reset_batch_list(self):
		self.batches = []
	
	def add_latent_batch(self, batch, dropout):
		self.batches.append((batch, dropout))

# This silences the annoying as hell Diffusers warning about the safety checker
@contextmanager
def override_print():
	original_print = sys.stdout
	sys.stdout = StringIO()
	yield
	sys.stdout = original_print

def save_sd1_pipeline(path, settings, accelerator, unet, text_model):
	if accelerator.is_main_process:
		with override_print():
			pipeline = StableDiffusionPipeline.from_pretrained(settings["model_name"], safety_checker=None)
			pipeline.unet = accelerator.unwrap_model(unet)
			if settings["train_text_encoder"]:
				pipeline.text_encoder = accelerator.unwrap_model(text_model)
			pipeline.save_pretrained(path)