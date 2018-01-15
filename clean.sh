#!/bin/bash

pkill -9 -f python
pkill -9 -f rogue

DIRECTORY=$(hostname)
cd /public/francesco_sovrano
rm -r $DIRECTORY
exit
