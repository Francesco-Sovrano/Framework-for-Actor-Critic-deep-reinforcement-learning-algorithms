import matplotlib.pyplot as plt
import datetime

def parse(log_fname):
    log = []
    with open(log_fname) as logfile:
        for i, line in enumerate(logfile):
            try:
                splitted = line.split(' ')
                date_str = splitted[0] + ' ' + splitted[1]
                date = datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S,%f')
                obj = {'date': date}
                for x in splitted[2:]:
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
	linewidth=0.4
	markersize=0.1
	x_label = 'Number of steps'
	
	logs = []
	for fname in logfiles:
		log = {'name': fname, 'data': parse(fname)}
		logs.append(log)

	shortest_log = None
	for data in logs:
		print(data['name'], 'len:', len(data['data']))
		if shortest_log is None or len(shortest_log['data']) > len(data['data']):
			shortest_log = data
	# print("shortest is:", shortest_log['name'], "with len:", len(shortest_log['data']))

	shortest_py_data = shortest_log['data']
	first_py_data = logs[0]['data']

	mean_steps = int(max_steps / len(shortest_py_data))
	x = [mean_steps*(i+1) for i in range(len(shortest_py_data))]
	for i, (key, y_label) in enumerate(
			[('accuracy', 'Success rate'), 
			 ('avg_reward', 'Average cumulative reward per episode'),
			 ('avg_tiles', 'Average number of seen tiles per episode'),
			 ('avg_success_steps', 'Average number of steps per won episode')]):
		ys = [[obj[key] for obj in data['data']] for data in logs]
		stats = [(data['name'], min(ys[i]), max(ys[i])) for i, data in enumerate(logs)]
		print(key, *stats, sep='\n')

		plot_args = []
		for y in ys:
			plot_args.extend([x, y[:len(shortest_py_data)], '-o'])
		plt.plot(*plot_args, linewidth=linewidth, markersize=markersize)
		plt.legend(['c2', 'c1'], markerscale=30)

		if i == 0:
			plt.ylim(0, 1)
			plt.yticks([0.1 * i for i in range(11)])

		plt.xticks([j*5e6 for j in range(7)], ["%sM" % (j*5) for j in range(7)])
		plt.xlabel(x_label)
		plt.ylabel(y_label)

		plt.savefig(figure_file)
		print("Plot figure saved in ", figure_file)