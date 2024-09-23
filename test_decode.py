import torch
# from diffusers import AutoencoderKL
from diffusers.utils.import_utils import is_xformers_available
from PIL import Image
import numpy as np

batch = torch.load("E:\\sd1_latents\\latent_cache_your_sd1_finetune_0.pt")

# Load the VAE model
# vae = AutoencoderKL.from_pretrained("X:\sd1-5", subfolder="vae")
# vae.to("cuda:0")
# vae.requires_grad_(False)
# vae.enable_slicing()
# if is_xformers_available():
# 	try:
# 		vae.enable_xformers_memory_efficient_attention()
# 	except Exception as e:
# 		print("no xformers")

# # Function to decode latents
# def decode_latents(latents):
# 	latents = 1 / 0.18215 * (latents.sample() * 0.18215)
# 	with torch.no_grad():
# 		image = vae.decode(latents).sample

# 	image = (image / 1).clamp(0, 1)
# 	image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
# 	images = (image * 255).round().astype("uint8")
# 	pil_images = [Image.fromarray(image) for image in images]
# 	return pil_images

# decoded_images = decode_latents(batch["vae_encoded"])
# decoded_images[0].save("decoded_image.png")

from transformers import CLIPTokenizer
tokenizer = CLIPTokenizer.from_pretrained("X:\sd1-5", subfolder="tokenizer")

for partial in batch["tokens"]:
	decoded_texts = tokenizer.batch_decode(partial, skip_special_tokens=True)
	pos = 0
	for text in decoded_texts:
		if text == "":
			print(pos, "EOS padded")
		else:
			print(pos, text)
		pos += 1
	print("")