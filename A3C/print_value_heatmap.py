# -*- coding: utf-8 -*-
import tensorflow as tf
from agent.server import Application
import argparse

parser = argparse.ArgumentParser(description='print value heatmap')
parser.add_argument('figure_file', help='the file in which to save the heatmap')
ARGS = parser.parse_args()
print("ARGS:", ARGS)

def main(argv):
	app = Application()
	app.print_value_heatmap(figure_file=ARGS.figure_file)

if __name__ == '__main__':
	tf.app.run()