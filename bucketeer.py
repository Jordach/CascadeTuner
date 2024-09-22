import torch
import torchvision
import numpy as np
from torchtools.transforms import SmartCrop
import math
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import warnings

class Bucketeer():
	def __init__(
		self,
		density=256*256,
		factor=8,
		ratios=[1/1, 1/2, 3/4, 3/5, 4/5, 6/9, 9/16],
		reverse_list=True,
		randomize_p=0.3,
		randomize_q=0.2,
		crop_mode='center',
		p_random_ratio=0.0,
		interpolate_nearest=False,
		transforms=None,
		settings=None
	):
		assert crop_mode in ['center', 'random', 'smart']
		self.crop_mode = crop_mode
		self.ratios = ratios
		if reverse_list:
			for r in list(ratios):
				if 1/r not in self.ratios:
					self.ratios.append(1/r)
		self.sizes = [(int(((density/r)**0.5//factor)*factor), int(((density*r)**0.5//factor)*factor)) for r in ratios]
		self.smartcrop = SmartCrop(int(density**0.5), randomize_p, randomize_q) if self.crop_mode=='smart' else None
		self.p_random_ratio = p_random_ratio
		self.interpolate_nearest = interpolate_nearest
		self.transforms = transforms
		self.density = density
		self.settings = settings
		self.factor = factor

	def get_closest_size(self, x, y):
		if self.p_random_ratio > 0 and np.random.rand() < self.p_random_ratio:
			best_size_idx = np.random.randint(len(self.ratios))
		else:
			w, h = x, y
			best_size_idx = np.argmin([abs(w/h-r) for r in self.ratios])
		return self.sizes[best_size_idx]

	def get_resize_size(self, orig_size, tgt_size):
		if (tgt_size[1]/tgt_size[0] - 1) * (orig_size[1]/orig_size[0] - 1) >= 0:
			alt_min = int(math.ceil(max(tgt_size)*min(orig_size)/max(orig_size)))
			resize_size = max(alt_min, min(tgt_size))
		else:
			alt_max = int(math.ceil(min(tgt_size)*max(orig_size)/min(orig_size)))
			resize_size = max(alt_max, max(tgt_size))
		return resize_size

	def load_and_resize(self, item, ratio):
		# Silences random warnings from PIL about "potential" DOS attacks
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			path = item
			image = Image.open(path).convert("RGB")
			w, h = image.size
			img_se = min(w, h)
			img_le = max(w, h)

			# Get crop and resizing info for the bucket's ratio
			resize_dims = self.get_closest_size(w, h)
			resize_se = min(resize_dims[0], resize_dims[1])
			resize_le = max(resize_dims[0], resize_dims[1])

			_crop_se = (math.sqrt(self.density)* 2)
			_crop_le = (math.sqrt(self.density)* 2) * ratio
			crop_dims = self.get_closest_size(_crop_se, _crop_le)
			crop_se = min(crop_dims[0], crop_dims[1])
			crop_le = max(crop_dims[0], crop_dims[1])

			# Get resizing factor
			scale_factor = (resize_se + 32) / img_se
			new_le = int(img_le * scale_factor)

			# A note on TorchVision CenterCrop and PIL resize:
			# They're H,W and not W,H oriented
			actual_ratio = w/h
			if actual_ratio >= 1:
				# size = [resize_se+32, new_le]
				size = [resize_se, resize_le]
				crop_size = [crop_se, crop_le]
			else:
				# size = [new_le, resize_se+32]
				size = [resize_le, resize_se]
				crop_size = [crop_le, crop_se]
				
			# resize_size = self.get_resize_size(img.shape[-2:], size)
			if self.interpolate_nearest:
				image = image.resize((size[1], size[0]), Image.Resampling.NEAREST)
				# img = torchvision.transforms.functional.resize(img, resize_size, interpolation=torchvision.transforms.InterpolationMode.NEAREST)
			else:
				image = image.resize((size[1], size[0]), Image.Resampling.LANCZOS)
				# img = torchvision.transforms.functional.resize(img, resize_size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True)
			# nw, nh = image.size
			img = self.transforms(image)
			del image
			
			if self.crop_mode == 'center':
				img = torchvision.transforms.functional.center_crop(img, crop_size)
			elif self.crop_mode == 'random':
				img = torchvision.transforms.RandomCrop(crop_size)(img)
			elif self.crop_mode == 'smart':
				self.smartcrop.output_size = crop_size
				img = self.smartcrop(img)
			else:
				img = torchvision.transforms.functional.center_crop(img, crop_size)
			
			# crop_img = img.shape[-2:]

			# file_path = f"{self.settings['checkpoint_path']}/{self.settings['experiment_id']}/dataset_debug.csv"
			# with open(file_path, "a") as f:
			# 	f.write(f"{actual_ratio:.2f},{w}x{h},{nw}x{nh},{crop_size[1]}x{crop_size[0]},{crop_img[1]}x{crop_img[0]}\n")
			return img

	def remove_duplicate_aspects(self):
		known_ratios = {}
		# <= 1
		for ratio in reversed(self.ratios):
			if ratio <= 1:
				pass
		
		# > 1
		for ratio in self.ratios:
			if ratio > 1:
				pass

	def test_resize(self, w, h, emit_print=False):
		# Get crop and resizing info for the bucket's ratio

		crop_dims = self.get_closest_size(w, h)
		resize_dims = self.get_resize_size((h, w), crop_dims)

		rs_se = min(resize_dims)
		rs_le = max(resize_dims)
		crop_se = min(crop_dims)
		crop_le = max(crop_dims)

		# A note on TorchVision CenterCrop and PIL resize:
		# They're H,W and not W,H oriented
		actual_ratio = w/h
		if actual_ratio >= 1:
			rs_size = [rs_le, rs_se]
			crop_size = [crop_le, crop_se]
		else:
			rs_size = [rs_se, rs_le]
			crop_size = [crop_se, crop_le]

		rs_w = rs_size[0]
		rs_h = rs_size[1]
		latent_w = crop_size[0] // self.factor
		latent_h = crop_size[1] // self.factor

		if emit_print:
			print(f"image in: {int(w)}x{int(h)}, resize: {rs_w}x{rs_h}, latent: {latent_w}x{latent_h}, ratio: {w/h:.2f}")
		return latent_w, latent_h