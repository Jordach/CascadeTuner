import json
from statistics import median

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

def lerp(a, b, factor):
	return a + (b - a) * factor

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
			# Also try and prevent pollution from low counts
			if t in tag_dict:
				if tag_dict[t] < settings["tag_weighting_count_low"]:
					mult = 1
				else:
					# Normalise to between 0-1
					mult = remap(
						tag_dict[t],
						settings["tag_weighting_count_low"],
						settings["tag_weighting_count_high"],
						0,
						1
					)
					# Clamp to prevent linear interpolation
					mult = clamp(mult, 0, 1)
					if settings["tag_weighting_exponent"] > 1:
						# Give it a curve to smooth lessen aggression of the mean
						mult = (1 - mult) ** settings["tag_weighting_exponent"]
					else:
						mult = mult * -1
					mult = clamp(mult, 0, 1)
					# Remap the 0-1 curve to the weighting range
					mult = remap(
						mult,
						0,
						1,
						settings["tag_weighting_multi_max"],
						settings["tag_weighting_multi_min"]
					)
				# Clamp to prevent linear interpolation
				mult = clamp(mult, settings["tag_weighting_multi_min"], settings["tag_weighting_multi_max"])
			else:
				mult = 1
			mults.append(mult)
	# This approximates EMA like weighting averaging
	median_mult = median(mults)
	mean_mult = list_average(mults)
	return lerp(median_mult, mean_mult, 0.8)