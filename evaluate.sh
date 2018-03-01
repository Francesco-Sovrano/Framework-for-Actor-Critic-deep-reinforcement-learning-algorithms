#!/bin/bash

DIRECTORY=$(hostname)

PYTHONPATH=Rogue/:A3C/$DIRECTORY python3 A3C/$DIRECTORY/evaluate/evaluate.py $*

