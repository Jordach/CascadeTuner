import re
import torch
import math
import random
import numpy as np

def shuffle_and_drop_tags(caption, settings, always_keep=None):
	tags = [tag.strip() for tag in caption.split(",")]
	random.shuffle(tags)

	# Determine which tags to always keep
	always_keep_tags = set()
	if always_keep:
		if isinstance(always_keep, str):
			always_keep_tags.add(always_keep)
		elif isinstance(always_keep, dict):
			always_keep_tags.update(always_keep.keys())
		elif isinstance(always_keep, (list, tuple, set)):
			always_keep_tags.update(always_keep)

	# Separate always-keep tags and other tags
	keep_tags = [tag for tag in tags if tag in always_keep_tags]
	other_tags = [tag for tag in tags if tag not in always_keep_tags]

	if "tag_dropout_percentage" in settings:
		# Only dropout if valid values are used
		num_tags = len(other_tags)
		num_kept = max(
			settings["tag_dropout_total_min"],
			int(num_tags * (1 - settings["tag_dropout_percentage"]))
		)

		if num_kept >= num_tags:
			kept_tags = other_tags
		else:
			kept_keys = np.random.choice(num_tags, num_kept, replace=False)
			kept_tags = [other_tags[i] for i in kept_keys]
	else:
		kept_tags = other_tags

	# Combine always-keep tags with the randomly kept tags
	final_tags = keep_tags + kept_tags
	random.shuffle(final_tags)

	return ", ".join(final_tags)

def tokenize_respecting_boundaries(tokenizer, captions, max_length=75):
	batch_size = len(captions)
	all_tokenized_captions = []
	all_attention_masks = []
	max_chunks = 0

	for caption in captions:
		words = re.findall(r'\s+|[^\s]+[.!?\'"\-)}\\]*', caption)
		tokenized_caption = []
		attention_mask = []
		current_chunk = []
		current_attention = []

		for word in words:
			word_tokens = tokenizer.encode(word.strip(), add_special_tokens=False)
			
			if len(current_chunk) + len(word_tokens) > max_length:
				# Pad and add the current chunk
				padded_chunk = current_chunk + [tokenizer.pad_token_id] * (max_length - len(current_chunk))
				tokenized_caption.append(padded_chunk)
				attention_mask.append(current_attention + [0] * (max_length - len(current_attention)))
				
				# Start a new chunk
				current_chunk = word_tokens
				current_attention = [1] * len(word_tokens)
			else:
				current_chunk.extend(word_tokens)
				current_attention.extend([1] * len(word_tokens))

			# Handle chunks that exceed max_length
			while len(current_chunk) >= max_length:
				tokenized_caption.append(current_chunk[:max_length])
				attention_mask.append([1] * max_length)
				current_chunk = current_chunk[max_length:]
				current_attention = current_attention[max_length:]

		# Add any remaining tokens in the last chunk
		if current_chunk:
			padded_chunk = current_chunk + [tokenizer.pad_token_id] * (max_length - len(current_chunk))
			tokenized_caption.append(padded_chunk)
			attention_mask.append(current_attention + [0] * (max_length - len(current_attention)))

		all_tokenized_captions.append(tokenized_caption)
		all_attention_masks.append(attention_mask)
		max_chunks = max(max_chunks, len(tokenized_caption))

	# Pad all captions to have the same number of chunks
	for i in range(batch_size):
		while len(all_tokenized_captions[i]) < max_chunks:
			all_tokenized_captions[i].append([tokenizer.pad_token_id] * max_length)
			all_attention_masks[i].append([0] * max_length)

	# Rearrange and convert to list of tensors
	tokenized_captions_list = []
	attention_masks_list = []

	for chunk_idx in range(max_chunks):
		chunk_tokens = [all_tokenized_captions[i][chunk_idx] for i in range(batch_size)]
		chunk_attention = [all_attention_masks[i][chunk_idx] for i in range(batch_size)]
		
		tokenized_captions_list.append(torch.tensor(chunk_tokens))
		attention_masks_list.append(torch.tensor(chunk_attention))

	return tokenized_captions_list, attention_masks_list

def get_text_embeds(dropout, text_model, accelerator, captions, att_mask, tokenizer, settings, batch_size, attn_mask, text_encoder_context):
	with text_encoder_context:
		text_embeddings = None
		text_embeddings_pool = None

		# Token concatenation things:
		max_length = tokenizer.model_max_length
		max_standard_tokens = max_length - 2
		token_chunks_limit = math.ceil(settings["max_token_limit"] / max_standard_tokens)

		if token_chunks_limit < 1:
			token_chunks_limit = 1

		if dropout:
			# Do not train the text encoder when getting empty embeds
			if settings["train_text_encoder"]:
				text_model.eval()
			captions_unpooled = ["" for _ in range(batch_size)]
			clip_tokens_unpooled = tokenizer(captions_unpooled, truncation=True, padding="max_length",
											max_length=tokenizer.model_max_length,
											return_tensors="pt").to(accelerator.device)

			text_encoder_output = accelerator.unwrap_model(text_model)(**clip_tokens_unpooled, output_hidden_states=True) if accelerator.num_processes > 1 else text_model(**clip_tokens_unpooled, output_hidden_states=True)
			text_embeddings = text_encoder_output.hidden_states[settings["clip_skip"]]
			text_embeddings_pool = text_encoder_output.text_embeds.unsqueeze(1) if "text_embeds" in text_encoder_output else None
			# Restore training mode for the text encoder
			if settings["train_text_encoder"]:
				text_model.train()
		else:
			for chunk_id in range(len(captions)):
				# Hard limit the tokens to fit in memory for the rare event that latent caches that somehow exceed the limit.
				if chunk_id > (token_chunks_limit):
					break

				token_chunk = captions[chunk_id].to(accelerator.device)
				token_chunk = torch.cat((torch.full((token_chunk.shape[0], 1), tokenizer.bos_token_id).to(accelerator.device), token_chunk, torch.full((token_chunk.shape[0], 1), tokenizer.eos_token_id).to(accelerator.device)), 1)
				attn_chunk = att_mask[chunk_id].to(accelerator.device)
				attn_chunk = torch.cat((torch.full((attn_chunk.shape[0], 1), 1).to(accelerator.device), attn_chunk, torch.full((attn_chunk.shape[0], 1), 1).to(accelerator.device)), 1)
				# First 75 tokens we allow BOS to not be masked - otherwise we mask them out
				fake_text_model = accelerator.unwrap_model(text_model) if accelerator.num_processes > 1 else text_model
				# encode
				inputs = {"input_ids": token_chunk}
				if attn_mask:
					inputs["attention_mask"] = attn_chunk
				text_encoder_output = fake_text_model(**inputs, output_hidden_states=True)
				hidden_states = text_encoder_output["hidden_states"][settings["clip_skip"]]
				hidden_states = hidden_states.to(dtype=torch.float32)
				text_embed = fake_text_model.text_model.final_layer_norm(hidden_states)

				if text_embeddings is None:
					text_embeddings = text_embed
					text_embeddings_pool = hidden_states.text_embeds.unsqueeze(1) if "text_embeds" in text_encoder_output else None
				else:
					text_embeddings = torch.cat((text_embeddings, text_embed), dim=-2)
					# text_embeddings_pool = torch.cat((text_embeddings_pool, text_encoder_output.text_embeds.unsqueeze(1)), dim=-2) if "text_embeds" in text_encoder_output else None

		return text_embeddings, text_embeddings_pool