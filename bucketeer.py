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
		transforms=None
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

			# Get crop for the bucket's ratio
			actual_ratio = w/h
			if actual_ratio <= 1:
				cw = (math.sqrt(self.density)* 2) * ratio
				ch = (math.sqrt(self.density)* 2)
			else:
				cw = (math.sqrt(self.density)* 2)
				ch = (math.sqrt(self.density)* 2) * ratio 
			size = self.get_closest_size(w, h)
			crop_size = self.get_closest_size(int(cw), int(ch))
			#resize_size = self.get_resize_size(img.shape[-2:], size)

			if self.interpolate_nearest:
				image = image.resize((size[0], size[1]), Image.Resampling.NEAREST)
				#img = torchvision.transforms.functional.resize(img, resize_size, interpolation=torchvision.transforms.InterpolationMode.NEAREST)
			else:
				image = image.resize((size[0], size[1]), Image.Resampling.LANCZOS)
				#img = torchvision.transforms.functional.resize(img, resize_size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True)
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
			
			return img