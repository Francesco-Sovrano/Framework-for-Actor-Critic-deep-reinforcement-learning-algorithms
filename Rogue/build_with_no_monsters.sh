#!/bin/bash

OLD_DIR="`pwd`"
MY_DIR="`dirname \"$0\"`"
MY_PATH="`realpath $MY_DIR`"
cd $MY_PATH

# apt-get install libncurses-dev
cd ./rogue5.4.4-ant-r1.1.4
./configure
make

cd $OLD_DIR