import json

def load_from_json_storage(path, tag_dict):
	with open(path, "r", encoding="utf-8") as data:
		data_lines = data.readlines()
		for dl in data_lines:
			tag = json.loads(dl)
			tag_dict[tag["tag"]] = tag["count"]

def save_to_json_storage(path, tag_dict):
	tags = tag_dict.keys()
	output_data = ""
	for tag in tags:
		data_str = json.dumps({"tag": tag, "count": tag_dict[tag]})
		output_data += f"{data_str}\n"
	
	with open(path, "w", encoding="utf-8") as file:
		file.write(output_data)

def add_tags_from_batch(tag_dict, captions):
	for caption in captions:
		tags = caption.split(",")
		for tag in tags:
			t = tag.strip()
			if t not in tag_dict:
				tag_dict[t] = 1
			else:
				tag_dict[t] += 1

def clamp(val, min_val, max_val):
	return max(min(val, max_val), min_val)

def remap(val, min_val, max_val, min_map, max_map):
	return min_map + (val - min_val) * (max_map - min_map) / (max_val - min_val)

def list_average(input):
	return sum(input) / len(input)

def get_loss_multiplier_for_batch(tag_dict, settings, captions):
	mults = []
	for caption in captions:
		tags = caption.split(",")
		for tag in tags:
			t = tag.strip()
			if t in tag_dict:
				# Normalise to between 0-1
				mult = remap(
					tag_dict[t],
					settings["tag_weighting_count_low"],
					settings["tag_weighting_count_high"],
					1,
					0
				)
				# Clamp to prevent linear interpolation
				mult = clamp(mult, settings["tag_weighting_count_low"], settings["tag_weighting_count_high"])
				# Give it a curve to smooth lessen aggression of the mean
				mult = mult ** 3
				# Remap the 0-1 curve to the weighting range
				mult = remap(
					tag_dict[t],
					0,
					1,
					settings["tag_weighting_multi_max"]
					settings["tag_weighting_multi_min"],
				)
				# Clamp to prevent linear interpolation
				mult = clamp(mult, settings["tag_weighting_multi_min"], settings["tag_weighting_multi_max"])
			else:
				mult = 1
			mults.append(mult)

	return list_average(mults)