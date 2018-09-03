# -*- coding: utf-8 -*-
import tensorflow as tf
import utils.plots as plt
import argparse

parser = argparse.ArgumentParser(description='draw plots')
parser.add_argument('output_file', nargs=1, help='the file in which to save the plots')
parser.add_argument('input_files', nargs='+', help='log files used to build the plot')
ARGS = parser.parse_args()
print("ARGS:", ARGS)

plt.plot_files(log_files=ARGS.input_files, figure_file=ARGS.output_file[0])