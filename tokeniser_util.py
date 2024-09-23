import re
import torch
import math

def tokenize_respecting_boundaries(tokenizer, captions):
	max_chunks = 0
	all_caption_chunks = []
	all_attention_chunks = []

	for caption in captions:
		caption_chunks = []
		attention_chunks = []
		current_chunk = []
		
		words = re.findall(r'\S+|\s+', caption)
		
		for word in words:
			word_tokens = tokenizer.encode(word.strip(), add_special_tokens=False)
			
			if len(current_chunk) + len(word_tokens) > 75:
				if current_chunk:
					padded_chunk = current_chunk + [tokenizer.pad_token_id] * (75 - len(current_chunk))
					caption_chunks.append(padded_chunk)
					attention_chunks.append([1] * len(current_chunk) + [0] * (75 - len(current_chunk)))
				current_chunk = word_tokens
			else:
				current_chunk.extend(word_tokens)
			
			while len(current_chunk) >= 75:
				caption_chunks.append(current_chunk[:75])
				attention_chunks.append([1] * 75)
				current_chunk = current_chunk[75:]
		
		if current_chunk:
			padded_chunk = current_chunk + [tokenizer.pad_token_id] * (75 - len(current_chunk))
			caption_chunks.append(padded_chunk)
			attention_chunks.append([1] * len(current_chunk) + [0] * (75 - len(current_chunk)))
		
		all_caption_chunks.append(caption_chunks)
		all_attention_chunks.append(attention_chunks)
		max_chunks = max(max_chunks, len(caption_chunks))

	# Pad all captions to have the same number of chunks
	for i in range(len(all_caption_chunks)):
		while len(all_caption_chunks[i]) < max_chunks:
			all_caption_chunks[i].append([tokenizer.pad_token_id] * 75)
			all_attention_chunks[i].append([0] * 75)

	# Convert to tensors and stack
	tokenized_captions = torch.tensor(all_caption_chunks)
	attention_masks = torch.tensor(all_attention_chunks)
	
	return tokenized_captions, attention_masks

def get_text_embeds(dropout, text_model, accelerator, captions, att_mask, tokenizer, settings, batch_size):
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
			text_encoder_output = accelerator.unwrap_model(text_model)(**{"input_ids": token_chunk, "attention_mask": attn_chunk}, output_hidden_states=True) if accelerator.num_processes > 1 else text_model(**{"input_ids": token_chunk, "attention_mask": attn_chunk}, output_hidden_states=True)

			if text_embeddings is None:
				text_embeddings = text_encoder_output["hidden_states"][settings["clip_skip"]]
				text_embeddings_pool = text_encoder_output.text_embeds.unsqueeze(1) if "text_embeds" in text_encoder_output else None
			else:
				text_embeddings = torch.cat((text_embeddings, text_encoder_output["hidden_states"][settings["clip_skip"]]), dim=-2)
				# text_embeddings_pool = torch.cat((text_embeddings_pool, text_encoder_output.text_embeds.unsqueeze(1)), dim=-2) if "text_embeds" in text_encoder_output else None

	return text_embeddings, text_embeddings_pool