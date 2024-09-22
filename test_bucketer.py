import torch
import torchvision
import numpy as np
from torchtools.transforms import SmartCrop
import math
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import warnings
from bucketeer import Bucketeer

test_ratios = []

r_min = 25
r_max = 400
cfactor = 8
base_res = 1024

for i in range(r_min, r_max):
	test_ratios.append(i/100)

bucketer = Bucketeer(
	density=base_res ** 2, # Image Size Square
	factor=cfactor, # VAE compression factor
	ratios=test_ratios, # Known aspect ratios
	p_random_ratio=0,
	transforms=torchvision.transforms.ToTensor(),
	settings = {}
)

for i in range(r_min, r_max):
	ratio = i/100
	if i < 1:
		bucketer.test_resize(1000*ratio, 1000, emit_print=True)
	else:
		bucketer.test_resize(1000, 1000*ratio, emit_print=True)