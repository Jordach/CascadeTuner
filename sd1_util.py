import torch
import random
import math
from torch.utils.data import Dataset
from tokeniser_util import tokenize_respecting_boundaries, pad_tokens

# This is known to work on multi-GPU setups
class SD1CachedLatents(Dataset):
	def __init__(self, accelerator, tokenizer=None, tag_shuffle=True):
		self.cache_paths = []
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

		cache = torch.load(self.cache_paths[index][0], map_location=self.accelerator.device)
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
			tokenized_captions, attention_masks = tokenize_respecting_boundaries(shuffled_captions)

			cache["tokens"] = tokenized_captions
			cache["att_mask"] = attention_masks

		return cache

	def get_batch_list(self):
		return self.batches
	
	def add_latent_batch(self, batch, dropout):
		self.batches.append((batch, dropout))