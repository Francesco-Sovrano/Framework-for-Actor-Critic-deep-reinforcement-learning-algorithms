#!/bin/bash

pkill -9 -f python
pkill -9 -f rogue

MY_PATH="`dirname \"$0\"`"
cd $MY_PATH
. .env/bin/activate

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

python3 ./A3C/train.py