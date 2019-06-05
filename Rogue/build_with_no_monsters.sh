#!/bin/bash

OLD_DIR="`pwd`"
MY_DIR="`dirname \"$0\"`"
cd $MY_DIR

chmod -R 777 rogue5.4.4-ant-r1.1.4
# apt-get install libncurses-dev
cd ./rogue5.4.4-ant-r1.1.4
./configure
make clean
make
chmod 777 rogue

cd $OLD_DIR