#!/bin/bash

DIRECTORY=$(hostname)

PYTHONPATH=A3C/$DIRECTORY python3 A3C/$DIRECTORY/checkpoint_conversion/conversion.py

