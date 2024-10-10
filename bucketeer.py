import torchvision
import numpy as np
from torchtools.transforms import SmartCrop
import math
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

	def clean_up_duplicate_buckets(self, emit_print=False):
		new_ratios = []
		latent_res = {}
		# Deduplicate using the fact that dicts are hashmaps
		for r in self.ratios:
			base_size = math.sqrt(self.density*2)
			if r < 1:
				lx, ly = self.test_resize(base_size, base_size/r, emit_print=False)
			else:
				lx, ly = self.test_resize(base_size*r, base_size, emit_print=False)
			latent_size = f"{int(lx)}x{int(ly)}"
			if latent_size not in latent_res:
				latent_res[latent_size] = True
				new_ratios.append(r)
			elif emit_print:
				print(f"Detected duplicate ratio: {r}, {latent_size}")
		
		# Finally
		self.ratios = new_ratios
		self.sizes = [(int(((self.density/r)**0.5//self.factor)*self.factor), int(((self.density*r)**0.5//self.factor)*self.factor)) for r in new_ratios]

	def get_closest_size(self, x, y):
		if self.p_random_ratio > 0 and np.random.rand() < self.p_random_ratio:
			best_size_idx = np.random.randint(len(self.ratios))
		else:
			w, h = x, y
			best_size_idx = np.argmin([abs(w/h-r) for r in self.ratios])
		
		size = self.sizes[best_size_idx]
		return (round(size[0] / self.factor) * self.factor, 
				round(size[1] / self.factor) * self.factor)

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
			actual_ratio = w/h
			img = self.transforms(image)
			del image

			# Get crop and resizing info for the bucket's ratio
			crop_size = self.get_closest_size(w, h)
			crop_size = (round(crop_size[0] / self.factor) * self.factor, 
						round(crop_size[1] / self.factor) * self.factor)
			
			resize_size = self.get_resize_size(img.shape[-2:], crop_size)
			resize_size = (round(resize_size / self.factor) * self.factor, 
						round(resize_size / self.factor) * self.factor)
			
			# Resize image
			img = torchvision.transforms.functional.resize(
				img, 
				resize_size, 
				interpolation=torchvision.transforms.InterpolationMode.BILINEAR, 
				antialias=True
			)
			
			# Crop image to target dimensions
			if self.crop_mode == 'center':
				img = torchvision.transforms.functional.center_crop(img, crop_size)
			elif self.crop_mode == 'random':
				img = torchvision.transforms.RandomCrop(crop_size)(img)
			elif self.crop_mode == 'smart':
				self.smartcrop.output_size = crop_size
				img = self.smartcrop(img)
			else:
				img = torchvision.transforms.functional.center_crop(img, crop_size)
			
			# Ensure final size is correct
			if img.shape[-2:] != crop_size:
				img = torchvision.transforms.functional.resize(
					img, 
					crop_size, 
					interpolation=torchvision.transforms.InterpolationMode.BILINEAR, 
					antialias=True
				)
			
			# file_path = f"{self.settings['checkpoint_path']}/{self.settings['experiment_id']}/dataset_debug.csv"
			# with open(file_path, "a") as f:
			# 	f.write(f"{actual_ratio:.2f},{w}x{h},{nw}x{nh},{crop_size[1]}x{crop_size[0]},{crop_img[1]}x{crop_img[0]}\n")
			return img

	def test_resize(self, w, h, emit_print=False):
		# Get crop and resizing info for the bucket's ratio
		actual_ratio = w/h

		crop_size = self.get_closest_size(w, h)
		crop_se = min(crop_size)
		crop_le = max(crop_size)
		
		resize_size = self.get_resize_size((h, w), crop_size)
		rs_se = resize_size
		rs_le = int(resize_size * actual_ratio) if actual_ratio >= 1 else int(resize_size / actual_ratio)

		# A note on TorchVision CenterCrop and PIL resize:
		# They're H,W and not W,H oriented
		if actual_ratio >= 1:
			crop_size = [crop_le, crop_se]
			rs_w = rs_le
			rs_h = rs_se
		else:
			crop_size = [crop_se, crop_le]
			rs_w = rs_se
			rs_h = rs_le

		latent_w = crop_size[0] / self.factor
		latent_h = crop_size[1] / self.factor

		if emit_print:
			print(f"image: {int(w)}x{int(h)}, resize: {rs_w}x{rs_h}, crop: {crop_size[0]}x{crop_size[1]}, latent: {latent_w}x{latent_h}, ratio: {actual_ratio:.4f}, resize ratio: {rs_w/rs_h:.4f}")
		return latent_w, latent_h

class StrictBucketeer:
	def __init__(
		self,
		density=256*256,
		factor=8,
		aspect_ratios=[],  # Pre-calculated w/h ratios
		crop_mode='center',
		transforms=None
	):
		assert crop_mode in ['center', 'random', 'smart']
		self.crop_mode = crop_mode
		self.density = density
		self.factor = factor
		
		# Generate bucket sizes and store in a dict
		self.buckets = self._generate_bucket_sizes(aspect_ratios)
		
		self.smartcrop = SmartCrop(int(density**0.5)) if self.crop_mode == 'smart' else None
		self.transforms = transforms

	def _generate_bucket_sizes(self, aspect_ratios):
		buckets = {}
		for ratio in aspect_ratios:
			if ratio <= 1:
				w = int(((self.density/ratio)**0.5//self.factor)*self.factor)
				h = int(((self.density*ratio)**0.5//self.factor)*self.factor)
			else:
				h = int(((self.density/ratio)**0.5//self.factor)*self.factor)
				w = int(((self.density*ratio)**0.5//self.factor)*self.factor)
			
			w = (w // self.factor) * self.factor
			h = (h // self.factor) * self.factor
			
			ratio_str = f"{ratio:.2f}"
			buckets[ratio_str] = (w, h)
		return buckets

	def get_resize_and_crop_sizes(self, w, h):
		aspect_ratio = w / h
		# ratio_str = f"{aspect_ratio:.2f}"
		
		closest_ratio = min(self.buckets.keys(), key=lambda x: abs(float(x) - aspect_ratio))
		target_size = self.buckets[closest_ratio]
		
		# Determine resize dimensions (resize smallest side to match target)
		if w <= h:
			resize_size = (target_size[0], int(h * target_size[0] / w))
		else:
			resize_size = (int(w * target_size[1] / h), target_size[1])
		
		return resize_size, target_size, closest_ratio

	def load_and_resize(self, item):
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			image = Image.open(item).convert("RGB")
			w, h = image.size
			img = self.transforms(image) if self.transforms else torchvision.transforms.ToTensor()(image)
			del image

			# Get the resize and crop sizes
			resize_size, crop_size, ratio = self.get_resize_and_crop_sizes(w, h)
			resize_size = min(resize_size)
			
			# Resize image
			img = torchvision.transforms.functional.resize(
				img, 
				resize_size,
				interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
				antialias=True
			)
			
			# Crop if necessary
			if img.shape[-2:] != crop_size:
				if self.crop_mode == 'center':
					img = torchvision.transforms.functional.center_crop(img, crop_size)
				elif self.crop_mode == 'random':
					img = torchvision.transforms.RandomCrop(crop_size)(img)
				elif self.crop_mode == 'smart':
					self.smartcrop.output_size = crop_size
					img = self.smartcrop(img)
			file_path = f"dataset_debug.csv"
			with open(file_path, "a") as f:
				f.write(f"{w/h:.2f},{ratio},{w}x{h},{crop_size[1]}x{crop_size[0]}\n")
			return img, ratio

	def __call__(self, item):
		return self.load_and_resize(item)