#!/bin/bash

MY_PATH="`dirname \"$0\"`"
cd $MY_PATH

# apt-get install libncurses-dev

cd ./rogue5.4.4-ant-r1.1.4_monsters
./configure
make clean
make -e EXTRA=-DSPAWN_MONSTERS
