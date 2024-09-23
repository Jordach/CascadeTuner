import argparse
import logging
import math
import os
import yaml
import numpy as np
import random

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm
from dataset_util import BucketWalker
from bucketeer import Bucketeer
from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer
from tokeniser_util import get_text_embeds
from sd1_util import SD1CachedLatents
from core_util import create_folder_if_necessary
logger = get_logger(__name__)

def vae_encode(images, vae):
    _images = images.to(dtype=vae.dtype)
    latents = vae.encode(_images).latent_dist.sample()
    return latents * 0.18215

def parse_args():
    parser = argparse.ArgumentParser(description="Custom training script for SD1.")
    parser.add_argument("--yaml", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--cache_only", default=False, action="store_true", help="Whether to quit after latent caching.")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # Load settings from YAML config
    with open(args.yaml, "r") as f:
        settings = yaml.safe_load(f)

    accelerator = Accelerator(
       gradient_accumulation_steps=settings["grad_accum_steps"],
       log_with="tensorboard",
       project_dir=f"{settings['checkpoint_path']}"
    )

    # Ensure output directory exists:
    if accelerator.is_main_process:
        if not os.path.exists(settings["checkpoint_path"]):
            os.makedirs(settings["checkpoint_path"])
    
    if settings["use_pytorch_cross_attention"]:
        accelerator.print("Activating efficient cross attentions.")
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    
    # Set up logging and seed
    logging.basicConfig(
        format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p", level=logging.INFO)
    set_seed(settings["seed"])

    # Load Tokeniser
    tokenizer = CLIPTokenizer.from_pretrained(settings["model_name"], subfolder="tokenizer")

    pre_dataset = []

    # Setup Dataloader Phase:
    if not settings["use_latent_cache"]:
        accelerator.print("Loading Dataset[s].")
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

        accelerator.print("Bucketing Info:")

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
    def latent_collate(batch):
        cache = torch.load(batch[0]["path"])
        if "dropout" in batch:
            cache[0]["dropout"] = True
        return cache
    
    # Load the VAE
    vae = AutoencoderKL.from_pretrained(settings["model_name"], subfolder="vae")
    vae.to(accelerator.device)
    vae.requires_grad_(False)
    vae.enable_slicing()
    if is_xformers_available() and args.attention=='xformers':
        try:
            vae.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning(
                "Could not enable memory efficient attention. Make sure xformers is installed"
                f" correctly and a GPU is available: {e}"
            )

    latent_cache = SD1CachedLatents(accelerator=accelerator, tokenizer=tokenizer, tag_shuffle=settings["tag_shuffling"])
    # Create a latent cache if we're not going to load an existing one:
    if settings["create_latent_cache"] and not settings["use_latent_cache"]:
        create_folder_if_necessary(settings["latent_cache_location"])
        step = 0
        for batch in tqdm(dataloader, desc="Latent Caching"):
            with torch.no_grad():
                batch["vae_encoded"] = vae_encode(batch["images"])
            del batch["images"]

            file_name = f"latent_cache_{settings['experiment_id']}_{step}.pt"
            torch.save(batch, os.path.join(settings["latent_cache_location"], file_name), False)
            latent_cache.add_latent_batch(os.path.join(settings["latent_cache_location"], file_name), False)
            step += 1
        if args.cache_only:
            return 0
    elif settings["use_latent_cache"]:
        # Load all latent caches from disk. Note that batch size is ignored here and can theoretically be mixed.
        if not os.path.exists(settings["latent_cache_location"]):
            raise Exception("Latent Cache folder does not exist. Please run latent caching first.")

        if len(os.listdir(settings["latent_cache_location"])) == 0:
            raise Exception("No latent caches to load. Please run latent caching first.")

        accelerator.print("Loading media from the Latent Cache.")
        for cache in os.listdir(settings["latent_cache_location"]):
            latent_path = os.path.join(settings["latent_cache_location"], cache)
            latent_cache.add_latent_batch(latent_path, False)
        
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

if __name__ == "__main__":
    main()