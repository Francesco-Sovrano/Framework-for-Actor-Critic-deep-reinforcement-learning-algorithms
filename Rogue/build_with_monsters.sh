#!/bin/bash

OLD_DIR="`pwd`"
MY_DIR="`dirname \"$0\"`"
cd $MY_DIR

chmod -R 777 rogue5.4.4-ant-r1.1.4_monsters
# apt-get install libncurses-dev
cd ./rogue5.4.4-ant-r1.1.4_monsters
./configure
make clean
make -e EXTRA=-DSPAWN_MONSTERS
chmod 777 rogue

cd $OLD_DIR