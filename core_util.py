# Single file import of Common utils from Stable Cascade

import os
import torch
import json
from pathlib import Path
import safetensors
import yaml
from torch import nn
import wandb
from munch import Munch
from abc import ABC, abstractmethod
import dataclasses
from dataclasses import dataclass, _MISSING_TYPE
from torch.utils.data import Dataset, DataLoader
from xformers_util import convert_state_dict_normal_attn_to_mha
import subprocess
from tqdm import tqdm

from torch.distributed import init_process_group, destroy_process_group, barrier
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType
)

EXPECTED = "___REQUIRED___"
EXPECTED_TRAIN = "___REQUIRED_TRAIN___"

# EMA
def update_weights_ema(tgt_model, src_model, beta=0.999):
    for self_params, src_params in zip(tgt_model.parameters(), src_model.parameters()):
        self_params.data = self_params.data * beta + src_params.data.clone().to(self_params.device) * (1-beta)
    for self_buffers, src_buffers in zip(tgt_model.buffers(), src_model.buffers()):
        self_buffers.data = self_buffers.data * beta + src_buffers.data.clone().to(self_buffers.device) * (1-beta)

# Save and Load
def create_folder_if_necessary(path):
    path = "/".join(path.split("/")[:-1])
    Path(path).mkdir(parents=True, exist_ok=True)

def safe_save(ckpt, path, iter, accelerator=None):
    '''
    try:
        os.remove(f"{path}.bak")
    except OSError:
        pass
    try:
        os.rename(path, f"{path}.bak")
    except OSError:
        pass
    '''
    if path.endswith(".pt") or path.endswith(".ckpt"):
        path = path.replace(".pt", f'-{iter}.pt')
        torch.save(ckpt, path)
    elif path.endswith(".json"):
        path = path.replace(".json", f'-{iter}.json')
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ckpt, f, indent=4)
    elif path.endswith(".safetensors"):
        path = path.replace(".safetensors", f'-{iter}.safetensors')
        #accelerator.save_model(ckpt, output_dir=path, safe_serialization=True)
        safetensors.torch.save_file(ckpt, path)
        tqdm.write(f"Saved model as: {path}")
    else:
        raise ValueError(f"File extension not supported: {path}")

def load_or_fail(path, wandb_run_id=None):
    accepted_extensions = [".pt", ".ckpt", ".json", ".safetensors"]
    try:
        assert any(
            [path.endswith(ext) for ext in accepted_extensions]
        ), f"Automatic loading not supported for this extension: {path}"
        if not os.path.exists(path):
            checkpoint = None
        elif path.endswith(".pt") or path.endswith(".ckpt"):
            checkpoint = torch.load(path, map_location="cpu")
        elif path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
        elif path.endswith(".safetensors"):
            checkpoint = {}
            with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    checkpoint[key] = f.get_tensor(key)
        return checkpoint
    except Exception as e:
        raise e

def load_optimizer(optim, optim_id=None, full_path=None, settings=None):
	if optim_id is not None and full_path is None:
		full_path = f"{settings['checkpoint_path']}/{settings['experiment_id']}/{optim_id}.pt"
	elif full_path is None and optim_id is None:
		raise ValueError(
			"This method expects either 'optim_id' or 'full_path' to be defined"
		)

	checkpoint = load_or_fail(full_path, wandb_run_id=None)
	if checkpoint is not None:
		try:
			optim.load_state_dict(checkpoint)
		# pylint: disable=broad-except
		except Exception as e:
			print("!!! Failed loading optimizer, skipping... Exception:", e)

	return optim

def save_optimizer(optim, optim_id=None, full_path=None, settings=None, accelerator=None, step=1):
	if optim_id is not None and full_path is None:
		full_path = f"{settings['checkpoint_path']}/{settings['experiment_id']}/{optim_id}.pt"
	elif full_path is None and optim_id is None:
		raise ValueError(
			"This method expects either 'optim_id' or 'full_path' to be defined"
		)
	create_folder_if_necessary(full_path)
	if accelerator.is_main_process:
		checkpoint = optim.state_dict()
		safe_save(checkpoint, full_path, step)
		del checkpoint

def save_model(model, model_id=None, full_path=None, accelerator=None, settings=None, step=1):
	if accelerator.is_main_process:
		if model_id is not None and full_path is None:
			full_path = f"{settings['checkpoint_path']}/{model_id}.{settings['checkpoint_extension']}"
		elif full_path is None and model_id is None:
			raise ValueError(
				"This method expects either 'model_id' or 'full_path' to be defined"
			)
		create_folder_if_necessary(full_path)
		checkpoint = model.state_dict()
		        
		if settings["flash_attention"]:
			checkpoint = convert_state_dict_normal_attn_to_mha(checkpoint)
                                        
		safe_save(checkpoint, full_path, step, accelerator=accelerator)
		del checkpoint

# Data
class MultiFilter():
    def __init__(self, rules, default=False):
        self.rules = rules
        self.default = default

    def __call__(self, x):
        try:
            x_json = x['json']
            if isinstance(x_json, bytes):
                x_json = json.loads(x_json) 
            validations = []
            for k, r in self.rules.items():
                if isinstance(k, tuple):
                    v = r(*[x_json[kv] for kv in k])
                else:
                    v = r(x_json[k])
                validations.append(v)
            return all(validations)
        except Exception:
            return False

class MultiGetter():
    def __init__(self, rules):
        self.rules = rules

    def __call__(self, x_json):
        if isinstance(x_json, bytes):
            x_json = json.loads(x_json) 
        outputs = []
        for k, r in self.rules.items():
            if isinstance(k, tuple):
                v = r(*[x_json[kv] for kv in k])
            else:
                v = r(x_json[k])
            outputs.append(v)
        if len(outputs) == 1:
            outputs = outputs[0]
        return outputs

def setup_webdataset_path(paths, cache_path=None):
    if cache_path is None or not os.path.exists(cache_path):
        tar_paths = []
        if isinstance(paths, str):
            paths = [paths]
        for path in paths:
            if path.strip().endswith(".tar"):
                # Avoid looking up s3 if we already have a tar file
                tar_paths.append(path)
                continue
            bucket = "/".join(path.split("/")[:3])
            result = subprocess.run([f"aws s3 ls {path} --recursive | awk '{{print $4}}'"], stdout=subprocess.PIPE, shell=True, check=True)
            files = result.stdout.decode('utf-8').split()
            files = [f"{bucket}/{f}" for f in files if f.endswith(".tar")]
            tar_paths += files

        with open(cache_path, 'w', encoding='utf-8') as outfile:
            yaml.dump(tar_paths, outfile, default_flow_style=False)
    else:
        with open(cache_path, 'r', encoding='utf-8') as file:
            tar_paths = yaml.safe_load(file)

    tar_paths_str = ",".join([f"{p}" for p in tar_paths])
    return f"pipe:aws s3 cp {{ {tar_paths_str} }} -"