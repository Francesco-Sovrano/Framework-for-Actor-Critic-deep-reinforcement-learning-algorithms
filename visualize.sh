#!/bin/bash

DIRECTORY=$(hostname)

PYTHONPATH=Rogue/:A3C/$DIRECTORY python A3C/$DIRECTORY/evaluate/visualize.py $*

