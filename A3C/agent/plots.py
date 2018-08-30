import matplotlib
matplotlib.use('Agg',force=True) # no display
from matplotlib import pyplot as plt
plt.ioff() # non-interactive back-end

import math
import datetime
import re
import numpy as np
import gc

import seaborn as sns # heatmaps
import imageio # GIFs
from PIL import Image, ImageFont, ImageDraw # images

import options
flags = options.get() # get command line args

def plot(logs, figure_file):
	# Get plot types
	stats = {}
	key_ids = {}
	for i in range(len(logs)):
		log = logs[i]
		# Get statistics keys
		if log["length"] < 2:
			continue
		(step, obj) = next(log["data"])
		log_keys = sorted(obj.keys()) # statistics keys sorted by name
		for key in log_keys:
			if key not in key_ids:
				key_ids[key] = len(key_ids)
		stats[i] = log_keys
	max_stats_count = len(key_ids)
	if max_stats_count <= 0:
		print("Not enough data for a reasonable plot")
		return
	# Create new figure and two subplots, sharing both axes
	ncols=3 if max_stats_count >= 3 else max_stats_count
	nrows=math.ceil(max_stats_count/ncols)
	figure, axes = plt.subplots(nrows=nrows, ncols=ncols, sharey=False, sharex=False, squeeze=False, figsize=(ncols*10,nrows*10))
	
	# Populate axes
	for log_id in range(len(logs)):
		log = logs[log_id]
		stat = stats[log_id]
		name = log["name"]
		data = log["data"]
		length = log["length"]
		if length < 2:
			print(name, " has not enough data for a reasonable plot")
			continue
		if length > flags.max_plot_size:
			plot_size = flags.max_plot_size
			data_per_plotpoint = length//plot_size
		else:
			plot_size = length
			data_per_plotpoint = 1
		# Build x, y
		x = {}
		y = {}
		for key in stat: # foreach statistic
			y[key] = {"min":float("+inf"), "max":float("-inf"), "data":[]}
			x[key] = []
		for _ in range(plot_size):
			value_sum = {}
			# initialize
			for key in stat: # foreach statistic
				value_sum[key] = 0
			# compute value_sum foreach key
			bad_obj_count = 0
			for _ in range(data_per_plotpoint):
				try:
					(step, obj) = next(data)
				except Exception as e:
					bad_obj_count += 1
					continue # try with next obj
				for key in stat: # foreach statistic
					v = obj[key]
					value_sum[key] += v
					if v > y[key]["max"]:
						y[key]["max"] = v
					if v < y[key]["min"]:
						y[key]["min"] = v
			if bad_obj_count < data_per_plotpoint:
				# add average to data for plotting
				for key in stat: # foreach statistic
					y[key]["data"].append(value_sum[key]/(data_per_plotpoint-bad_obj_count))
					x[key].append(step)
		# Populate axes
		print(name)
		for j in range(ncols):
			for i in range(nrows):
				idx = j if nrows == 1 else i*ncols+j
				if idx >= len(stat):
					continue
				key = stat[idx]
				ax_id = key_ids[key]
				ax = axes[ax_id//ncols][ax_id%ncols]
				# print stats
				print("    ", y[key]["min"], " < ", key, " < ", y[key]["max"])
				# ax
				ax.set_ylabel(key)
				ax.set_xlabel('step')
				# ax.plot(x, y, linewidth=linewidth, markersize=markersize)
				ax.plot(x[key], y[key]["data"], label=name)
				ax.legend()
				ax.grid(True)
	# remove unused axes
	for ax_id in range(len(key_ids), nrows*ncols):
		ax = axes[ax_id//ncols][ax_id%ncols] # always nrows > 1
		figure.delaxes(ax)
		
	figure.savefig(figure_file)
	print("Plot figure saved in ", figure_file)
	# Release memory
	figure.clear()
	# for ax_id in range(nrows*ncols):
		# axes[ax_id//ncols][ax_id%ncols].cla()
	plt.close(figure)
	gc.collect()

def plot_files(log_files, figure_file):
	logs = []
	for fname in log_files:
		logs.append({'name': fname, 'data': parse(fname), 'length':get_length(fname)})
	plot(logs, figure_file)
	
def get_length(file):
	return sum(1 for line in open(file))
	
def parse(log_fname):
	with open(log_fname, 'r') as logfile:
		for i, line in enumerate(logfile):
			try:
				splitted = line.split(' ')
				# date_str = splitted[0] + ' ' + splitted[1]
				# date = datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S,%f')
				# obj = {'date': date}
				# Get step
				if '<' in splitted[2]:
					step = re.sub('[<>]', '', splitted[2]) # remove following chars: <>
					step = int(step)
					xs = splitted[3:]
				else:
					step = i
					xs = splitted[2:]
				# Get objects
				obj = {}
				for x in xs:
					x = re.sub('[\',\[\]]', '', x) # remove following chars: ',[]
					# print(x)
					key, val = x.split('=')
					obj[key] = float(val)
				# print (obj)
				yield (step, obj)
			except Exception as e:
				print("exc %s on line %s" % (repr(e), i+1))
				print("skipping line")
				continue
	
def heatmap(heatmap, figure_file):
	figure, ax = plt.subplots(nrows=1, ncols=1)
	sns.heatmap(data=heatmap, ax=ax)
	figure.savefig(figure_file)
	# Release memory
	figure.clear()
	# ax.cla()
	plt.close(figure)
	gc.collect()
	
def ascii_image(string, file_name):
	# find image size
	font = ImageFont.load_default()
	splitlines = string.splitlines()
	text_width = 0
	text_height = 0
	for line in splitlines:
		text_size = font.getsize(line) # for efficiency's sake, split only on the first newline, discard the rest
		text_width = max(text_width,text_size[0])
		text_height += text_size[1]+5
	text_width += 10
	# create image
	source_img = Image.new('RGB', (text_width,text_height), "black")
	draw = ImageDraw.Draw(source_img)
	draw.text((5, 5), string, font=font)
	source_img.save(file_name, "JPEG")
	
def combine_images(images_list, file_name):
	imgs = [ Image.open(i) for i in images_list ]
	# pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
	min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
	imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
	# save the picture
	imgs_comb = Image.fromarray( imgs_comb )
	imgs_comb.save( file_name )
	
def rgb_array_image(array, file_name):
	img = Image.fromarray(array, 'RGB')
	img.save(file_name)
	
def make_gif(gif_path, file_list):
	with imageio.get_writer(gif_path, mode='I', duration=flags.gif_speed) as writer:
		for filename in file_list:
			image = imageio.imread(filename)
			writer.append_data(image)