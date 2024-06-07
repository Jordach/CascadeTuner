# Custom dataloader implementation for Stable Cascade, based on StableTuner's
import warnings
import os
import random
import math
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFile
from PIL import UnidentifiedImageError

class BucketWalker():
	def __init__(
		self,
		reject_aspects=1000,
		path=None,
		tokenizer=None
	):
		self.images = []
		self.reject_aspects=reject_aspects
		self.reject_count = 0
		self.interrupted = False
		self.final_dataset = []
		self.buckets = {}
		self.tokenizer = tokenizer

		# Optionally provide a path so you can manually walk folders later.
		if path is not None:
			self.walk_dataset_folders(self, path)

	def bucketize(self, batch_size):
		all_aspects = self.buckets.keys()
		# Make all buckets divisible by batch size
		original_count = 0
		for aspect in all_aspects:
			aspect_len = len(self.buckets[aspect])
			original_count += aspect_len
			if aspect_len > 0:
				remain = batch_size - aspect_len % batch_size
				if remain > 0 and remain != batch_size:
					for i in range(remain):
						self.buckets[aspect].extend(random.sample(self.buckets[aspect], 1))
					print(f"Bucket {aspect} has {aspect_len} images, duplicated {remain} images to fit batch size.")
				else:
					print(f"Bucket {aspect} has {aspect_len} images, duplicates not required, nice!")
				random.shuffle(self.buckets[aspect])
				# Finally
				self.final_dataset.extend(self.buckets[aspect])
		
		total_count = len(self.final_dataset)
		print(f"Original Image Count: {original_count}")
		print(f"Total Image Count:    {total_count}")
		print(f"Total Step Count:     {total_count // batch_size}")

	@staticmethod
	def walk_dataset_folders(self, path):
		if self.interrupted:
			return
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			sub_dirs = []
			path_list = os.listdir(path)
			pbar = tqdm(path_list, desc=f"* Processing: {path}")
			for f in path_list:
				current = os.path.join(path, f)
				if os.path.isfile(current):
					ext = os.path.splitext(f)[1].lower()
					if ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
						try:
							# Converting to RGB will ensure no truncated or malformed images pass into the training set
							image = Image.open(current).convert("RGB")
							width, height = image.size
							aspect = width / height
							# Only allow aspect ratios above this, a ratio of 6 would allow all aspects less than 1:6
							# The image is always oriented "vertically" for this check to ensure consistency against all cases.
							se = min(width, height)
							le = max(width, height)
							alt_aspect = le/se
							if alt_aspect <= self.reject_aspects:
								trimmed_aspect = f"{aspect:.2f}"
								txt_file = os.path.splitext(current)[0] + ".txt"
								caption = ""
								if os.path.exists(txt_file):
									with open(txt_file, "r", encoding="utf-8") as txt:
										caption = txt.readline().strip()
										if len(caption) < 1:
											raise ValueError(f"Could not find valid text in: {txt_file}")
										#Should it succeed in finding captions add it:
										file_dict = {"path": current, "width": width, "height": height, "aspect": trimmed_aspect, "caption": caption}
										if trimmed_aspect not in self.buckets:
											self.buckets[trimmed_aspect] = []
										self.buckets[trimmed_aspect].append(file_dict)
								else:
									tqdm.write(f"No text file found for: {txt_file}; skipping.")
									self.reject_count += 1
						except UnidentifiedImageError as e:
							tqdm.write(f"Cannot load {current}, file may be broken or corrupt.")
							self.reject_count += 1
						except Image.DecompressionBombWarning as e:
							tqdm.write(f"Cannot load {current}, file is too large.")
							self.reject_count += 1
						except ValueError as e:
							tqdm.write("Cannot load {current}, file is either too large or a ValueError occurred.")
							self.reject_count += 1
						except KeyboardInterrupt:
							self.interrupted = True
						except:
							tqdm.write(f"Cannot load {current}, file may be broken or corrupt.")
							self.reject_count += 1
				if os.path.isdir(current):
					sub_dirs.append(current)
				pbar.update(1)

			for dir in sub_dirs:
				self.walk_dataset_folders(self=self, path=dir)

	def scan_folder(self, path):
		self.walk_dataset_folders(self, path)
	
	def get_rejects(self):
		return self.reject_count

	def get_buckets(self):
		# Deduplicate buckets with a > 1 aspect ratio
		aspects = self.buckets.keys()
		buckets = {}
		
		for aspect in aspects:
			actual_aspect = aspect
			float_aspect = float(aspect)
			# Also convert landscape/horizontal aspects to vertical ones
			if float_aspect > 1:
				actual_aspect = f"{1/float_aspect:.2f}"
			
			if actual_aspect not in buckets:
				buckets[actual_aspect] = True

		all_aspects = buckets.keys()
		output_buckets = []
		for aspect in all_aspects:
			output_buckets.append(float(aspect))

		return output_buckets

	def get_final_dataset(self):
		return self.final_dataset

	def __len__(self):
		return len(self.final_dataset)

	def __getitem__(self, i):
		idx = i % len(self.final_dataset)

		item = self.final_dataset[idx]
		tokens = self.tokenizer(
			item["caption"],
			padding=False,
			add_special_tokens=False,
			verbose=False
		).input_ids
		
		return {"images": item["path"], "caption": item["caption"], "tokens": tokens, "aspects": item["aspect"]}

from torch.utils.data import Dataset
import torch

# This is known to work on multi-GPU setups
class CachedLatents(Dataset):
	def __init__(self, accelerator, tokenizer=None, tag_shuffle=False, retokenise=False):
		self.cache_paths = []
		self.accelerator = accelerator
		self.tokenizer = tokenizer
		self.tag_shuffle = tag_shuffle
		self.retokenise = retokenise
		if tag_shuffle:
			print("Will shuffle captions in Latent Caches.")

	def __len__(self):
		return len(self.cache_paths)

	def __getitem__(self, index):
		if index == 0:
			random.shuffle(self.cache_paths)

		cache = torch.load(self.cache_paths[index][0], map_location=self.accelerator.device)
		if self.cache_paths[index][1]:
			cache["dropout"] = True
		if self.tag_shuffle:
			del cache["tokens"]
			del cache["att_mask"]

			shuffled_captions = []
			for caption in cache["captions"]:
				shuffled_caption = ""
				tags = caption.split(",")
				random.shuffle(tags)
				for tag in tags:
					t = tag.strip()
					shuffled_caption += f"{t}, "
				shuffled_captions.append(shuffled_caption)

			raw_tokens = self.tokenizer(
				shuffled_captions,
				padding=False,
				add_special_tokens=False,
				verbose=False
			).input_ids

			# Get total number of chunks
			max_len = max(len(x) for x in raw_tokens)
			num_chunks = math.ceil(max_len / (self.tokenizer.model_max_length - 2))
			if num_chunks < 1:
				num_chunks = 1
			
			# Get the true padded length of the tokens
			len_input = self.tokenizer.model_max_length - 2
			if num_chunks > 1:
				len_input = (self.tokenizer.model_max_length * num_chunks) - (num_chunks * 2)
			
			# Tokenize!
			tokens = self.tokenizer.pad(
				{"input_ids": raw_tokens},
				padding="max_length",
				max_length=len_input,
				return_tensors="pt",
			).to("cpu")
			b_tokens = tokens["input_ids"]
			b_att_mask = tokens["attention_mask"]

			max_standard_tokens = self.tokenizer.model_max_length - 2
			true_len = max(len(x) for x in b_tokens)
			n_chunks = np.ceil(true_len / max_standard_tokens).astype(int)
			max_len = n_chunks.item() * max_standard_tokens

			# Properly manage memory here - don't bother loading tokens onto GPU.
			# Should prevent an OOM scenario on the GPU.
			cropped_tokens = [b_tokens[:, i:i + max_standard_tokens].clone().detach().to("cpu") for i in range(0, max_len, max_standard_tokens)]
			cropped_attn = [b_att_mask[:, i:i + max_standard_tokens].clone().detach().to("cpu") for i in range(0, max_len, max_standard_tokens)]
			
			cache["tokens"] = cropped_tokens
			cache["att_mask"] = cropped_attn

			del tokens
			del b_tokens
			del b_att_mask
		if self.retokenise:
			del cache["tokens"]
			del cache["att_mask"]
			
			raw_tokens = self.tokenizer(
				cache["captions"],
				padding=False,
				add_special_tokens=False,
				verbose=False
			).input_ids

			# Get total number of chunks
			max_len = max(len(x) for x in raw_tokens)
			num_chunks = math.ceil(max_len / (self.tokenizer.model_max_length - 2))
			if num_chunks < 1:
				num_chunks = 1
			
			# Get the true padded length of the tokens
			len_input = self.tokenizer.model_max_length - 2
			if num_chunks > 1:
				len_input = (self.tokenizer.model_max_length * num_chunks) - (num_chunks * 2)
			
			# Tokenize!
			tokens = self.tokenizer.pad(
				{"input_ids": raw_tokens},
				padding="max_length",
				max_length=len_input,
				return_tensors="pt",
			).to("cpu")
			b_tokens = tokens["input_ids"]
			b_att_mask = tokens["attention_mask"]

			max_standard_tokens = self.tokenizer.model_max_length - 2
			true_len = max(len(x) for x in b_tokens)
			n_chunks = np.ceil(true_len / max_standard_tokens).astype(int)
			max_len = n_chunks.item() * max_standard_tokens

			# Properly manage memory here - don't bother loading tokens onto GPU.
			# Should prevent an OOM scenario on the GPU.
			cropped_tokens = [b_tokens[:, i:i + max_standard_tokens].clone().detach().to("cpu") for i in range(0, max_len, max_standard_tokens)]
			cropped_attn = [b_att_mask[:, i:i + max_standard_tokens].clone().detach().to("cpu") for i in range(0, max_len, max_standard_tokens)]
			
			cache["tokens"] = cropped_tokens
			cache["att_mask"] = cropped_attn

			del tokens
			del b_tokens
			del b_att_mask

		return cache

	def get_cache_list(self):
		return self.cache_paths

	def add_cache_location(self, cache_path, dropout):
		self.cache_paths.append((cache_path, dropout))

# Work in progress
class RegularLatents(Dataset):
	def __init__(self, bucketer, accelerator):
		self.batches = []
		self.bucketer = bucketer
		self.accelerator = accelerator

	def __len__(self):
		return len(self.batches)

	def __getitem__(self, index):
		if index == 0:
			random.shuffle(self.batches)

		images = []
		for i in range(0, len(self.batches[index][0]["images"])):
			images.append(self.bucketer.load_and_resize(self.batches[index][0][0]["images"][i]), float(self.batches[index][0][0]["aspect"][i]))
		images = torch.stack(images)
		images = images.to(memory_format=torch.contiguous_format)
		images = images.to(self.accelerator.device)

		batch = {
			"aspects": self.batches[0][0]["aspects"],
			"images": images,
			"tokens": None,
			"att_mask": None,
			"captions": self.batches[0][0]["captions"],
			"dropout": False
		}
		
		return batch

	def get_batch_list(self):
		return self.batches
	
	def add_latent_batch(self, batch, dropout):
		self.batches.append((batch, dropout))