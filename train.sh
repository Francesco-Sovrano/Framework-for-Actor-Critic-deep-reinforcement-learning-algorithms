#!/bin/bash

DIRECTORY=$(hostname)
cd /public/francesco_sovrano
if [ ! -d "$DIRECTORY" ]; then
  mkdir $DIRECTORY
fi
. .env/bin/activate
cd $DIRECTORY
/home/students/francesco.sovrano/Documents/ML/A3C/train.sh > out.log &
disown
exit
