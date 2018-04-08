
import argparse
import matplotlib.pyplot as plt
import log_parser


parser = argparse.ArgumentParser(description='draw plots')
parser.add_argument('maxsteps', type=int, help='number of training steps on the x-axis (this is different from log lines)')
parser.add_argument('logfile', nargs='+', help='log files')
ARGS = parser.parse_args()
print("ARGS:", ARGS)

linewidth=0.4
markersize=0.1
max_steps = ARGS.maxsteps
x_label = 'Number of steps'

logs = []
for fname in ARGS.logfile:
    log = {'name': fname, 'data': log_parser.parse(fname)}
    logs.append(log)

shortest_log = None
for data in logs:
    print(data['name'], 'len:', len(data['data']))
    if shortest_log is None or len(shortest_log['data']) > len(data['data']):
        shortest_log = data

print("shortest is:", shortest_log['name'], "with len:", len(shortest_log['data']))

shortest_py_data = shortest_log['data']

first_py_data = logs[0]['data']

mean_steps = int(max_steps / len(shortest_py_data))
x = [mean_steps*(i+1) for i in range(len(shortest_py_data))]


for i, (key, y_label) in enumerate(
        [('win_perc', 'Success rate'), ('reward_avg', 'Average cumulative reward per episode'),
         ('tiles_avg', 'Average number of seen tiles per episode'),
         ('steps_avg', 'Average number of steps per won episode')]):
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

    if i < 3:
        plt.figure()
    else:
        plt.show()
