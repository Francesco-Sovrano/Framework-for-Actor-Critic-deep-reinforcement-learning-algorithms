import sys
import math
import datetime
import re

import matplotlib
matplotlib.use('Agg') # non-interactive backend
import matplotlib.pyplot as plt

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

def plot(max_steps, logfiles, figure_file):
	# linewidth=0.4
	# markersize=0.1
	logs = []
	min_data_length = sys.maxsize # max int
	for fname in logfiles:
		log = {'name': fname, 'data': parse(fname)}
		if len(log["data"]) < min_data_length:
			min_data_length = len(log["data"])
		logs.append(log)
	stats = sorted(logs[0]["data"][0].keys(), key=lambda t: t[0]) # statistics keys sorted by name
	
	# Create new figure and two subplots, sharing both axes
	ncols=3
	nrows=math.ceil(len(stats)/ncols)
	figure, plots = plt.subplots(nrows=nrows, ncols=ncols, sharey=False, sharex=False, figsize=(ncols*10,nrows*10))
	
	x = range(min_data_length)
	x_label = 'Number of steps'
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
				for obj in data[:min_data_length]:
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