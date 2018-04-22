#!/bin/bash

DIRECTORY=$(hostname)

PYTHONPATH=./:A3C/$DIRECTORY:A3C/$DIRECTORY/evaluate python3 A3C/$DIRECTORY/evaluate/evaluate.py $*

