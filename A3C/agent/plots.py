import sys
import math
import datetime
import re

import matplotlib
matplotlib.use('Agg') # non-interactive backend
import matplotlib.pyplot as plt

# get command line args
import options
flags = options.get()

def plot(logs, figure_file):
	# Find the smallest log file, its length is the maximum length of each plot of each log
	min_data_length = sys.maxsize # max int
	for log in logs:
		if len(log["data"]) < min_data_length:
			min_data_length = len(log["data"])
	# Get statistics keys
	stats = sorted(logs[0]["data"][0].keys(), key=lambda t: t[0]) # statistics keys sorted by name	
	# Create new figure and two subplots, sharing both axes
	ncols=3
	nrows=math.ceil(len(stats)/ncols)
	figure, plots = plt.subplots(nrows=nrows, ncols=ncols, sharey=False, sharex=False, figsize=(ncols*10,nrows*10))
	# Populate plots
	data_length = min_data_length-flags.match_count_for_evaluation
	if data_length < 2:
		print("Not enough data for a reasonable plot")
		return
		
	x = range(data_length)
	x_label = 'Episodes'
	for log in logs:
		name = log["name"]
		data = log["data"]
		for j in range(ncols):
			for i in range(nrows):
				plot = plots[i][j]
				idx = i*ncols+j
				if idx >= len(stats):
					figure.delaxes(plot) # remove unused plot
					continue
				key = stats[idx]
				# build y
				y = []
				min = float("+inf")
				max = float("-inf")
				for obj in data[flags.match_count_for_evaluation:min_data_length]:
					val = obj[key]
					y.append(val)
					if val > max:
						max = val
					if val < min:
						min = val
				# print stats
				print(min, " < ", key, " < ", max)
				# plot
				plot.set_ylabel(key)
				# plot.plot(x, y, linewidth=linewidth, markersize=markersize)
				plot.plot(x, y)
				plot.grid(True)

	plt.legend([log["name"] for log in logs], markerscale=30, loc='upper center', bbox_to_anchor=[0.5, -0.05])
	plt.xlabel(x_label)
	figure.savefig(figure_file)
	print("Plot figure saved in ", figure_file)

def plot_files(log_files, figure_file):
	logs = []
	for fname in logfiles:
		logs.append({'name': fname, 'data': parse(fname)})
	plot(logs, figure_file)
	
def parse(log_fname):
	log = []
	with open(log_fname) as logfile:
		for i, line in enumerate(logfile):
			try:
				splitted = line.split(' ')
				# date_str = splitted[0] + ' ' + splitted[1]
				# date = datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S,%f')
				# obj = {'date': date}
				obj = {}
				for x in splitted[2:]:
					x = re.sub('[\',\[\]]', '', x)
					# print(x)
					if '=' in x:
						key, val = x.split('=')
						obj[key] = float(val)
				log.append(obj)
			except Exception as e:
				print("exc %s on line %s" % (repr(e), i+1))
				print("skipping line")
				continue
	return log