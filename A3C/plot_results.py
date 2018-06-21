# -*- coding: utf-8 -*-
import agent.plots as plt
import argparse

parser = argparse.ArgumentParser(description='draw plots')
parser.add_argument('figure_file', nargs='+', help='the file in which to save the plots')
parser.add_argument('log_files', nargs='+', help='log files used to build the plot')
ARGS = parser.parse_args()
print("ARGS:", ARGS)

plt.plot_files(log_files=ARGS.log_files, figure_file=ARGS.figure_file)