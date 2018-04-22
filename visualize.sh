#!/bin/bash

DIRECTORY=$(hostname)

PYTHONPATH=./:A3C/$DIRECTORY:A3C/$DIRECTORY/evaluate python A3C/$DIRECTORY/evaluate/visualize.py $*

