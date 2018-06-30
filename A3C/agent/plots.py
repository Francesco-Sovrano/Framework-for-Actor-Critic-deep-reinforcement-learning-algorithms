import sys
import math
import datetime
import re

import matplotlib
matplotlib.use('Agg') # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# get command line args
import options
flags = options.get()

def plot(logs, figure_file):
	# Find the smallest log file, its length is the maximum length of each plot of each log
	min_data_length = sys.maxsize # max int
	for log in logs:
		if log["length"] < min_data_length:
			min_data_length = log["length"]
	if min_data_length < 2:
		print("Not enough data for a reasonable plot")
		return		
	# Get statistics keys
	stats = sorted(next(logs[0]["data"]).keys(), key=lambda t: t[0]) # statistics keys sorted by name
	stats_count = len(stats)
	# Create new figure and two subplots, sharing both axes
	ncols=3 if stats_count >= 3 else stats_count
	nrows=math.ceil(stats_count/ncols)
	figure, plots = plt.subplots(nrows=nrows, ncols=ncols, sharey=False, sharex=False, figsize=(ncols*10,nrows*10))
	# Populate plots
	if min_data_length > flags.max_plot_size:		
		plot_size = flags.max_plot_size
		data_per_plotpoint = min_data_length//plot_size
	else:
		plot_size = min_data_length
		data_per_plotpoint = 1
		
	for log in logs:
		name = log["name"]
		data = log["data"]
		# Build x
		x = [i*data_per_plotpoint for i in range(plot_size)]
		# Build y
		y = {}
		for key in stats: # foreach statistic
			y[key] = {"min":float("+inf"), "max":float("-inf"), "data":[]}
		for _ in range(plot_size):
			value_sum = {}
			# initialize
			for key in stats: # foreach statistic
				value_sum[key] = 0
			# compute value_sum foreach key
			bad_obj_count = 0
			for _ in range(data_per_plotpoint):
				try:
					obj = next(data)
				except Exception as e:
					bad_obj_count += 1
					continue # try with next obj
				for key in stats: # foreach statistic
					v = obj[key]
					value_sum[key] += v
					if v > y[key]["max"]:
						y[key]["max"] = v
					if v < y[key]["min"]:
						y[key]["min"] = v
			if bad_obj_count == data_per_plotpoint:
				x.pop() # remove an element from x
			else:
				# add average to data for plotting
				for key in stats: # foreach statistic
					y[key]["data"].append(value_sum[key]/(data_per_plotpoint-bad_obj_count))
		# Populate plots
		for j in range(ncols):
			for i in range(nrows):
				if nrows == 1:
					plot = plots[j]
					idx = j
				else:
					plot = plots[i][j]
					idx = i*ncols+j
				if idx >= stats_count:
					figure.delaxes(plot) # remove unused plot
					continue
				key = stats[idx]
				# print stats
				print(y[key]["min"], " < ", key, " < ", y[key]["max"])
				# plot
				plot.set_ylabel(key)
				# plot.plot(x, y, linewidth=linewidth, markersize=markersize)
				plot.plot(x, y[key]["data"])
				plot.grid(True)

	plt.legend([log["name"] for log in logs], markerscale=30, loc='upper center', bbox_to_anchor=[0.5, -0.05])
	plt.xlabel('Episodes')
	figure.savefig(figure_file)
	print("Plot figure saved in ", figure_file)
	figure.clf() # release memory
	plt.close() # release memory

def plot_files(log_files, figure_file):
	logs = []
	for fname in log_files:
		logs.append({'name': fname, 'data': parse(fname), 'length':get_length(fname)})
	plot(logs, figure_file)
	
def get_length(file):
	return sum(1 for line in open(file))
	
def parse(log_fname):
	logfile = open(log_fname, 'r')
	for i, line in enumerate(logfile):
		try:
			splitted = line.split(' ')
			# date_str = splitted[0] + ' ' + splitted[1]
			# date = datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S,%f')
			# obj = {'date': date}
			obj = {}
			for x in splitted[2:]:
				x = re.sub('[\',\[\]]', '', x) # remove following chars: ',[]
				# print(x)
				key, val = x.split('=')
				obj[key] = float(val)
			# print (obj)
			yield obj
		except Exception as e:
			print("exc %s on line %s" % (repr(e), i+1))
			print("skipping line")
			continue
	logfile.close()
	
def heatmap(map, figure_file):
	ax = sns.heatmap(map)
	plt.show()
	plt.savefig(figure_file)
	plt.close()