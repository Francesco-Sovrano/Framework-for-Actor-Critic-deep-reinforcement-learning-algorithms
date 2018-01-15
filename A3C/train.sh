#!/bin/bash

pkill -9 -f python
pkill -9 -f rogue
# rm -r log
if [ ! -d "log" ]; then
  mkdir log
fi
cd ./log
if [ ! -d "screenshots" ]; then
  mkdir screenshots
fi
if [ ! -d "performance" ]; then
  mkdir performance
fi
cd ..
python3 ./A3C/benes/train.py