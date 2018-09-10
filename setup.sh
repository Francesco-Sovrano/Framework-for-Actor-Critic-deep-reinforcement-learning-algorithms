#!/bin/bash

MY_PATH="`dirname \"$0\"`"
cd $MY_PATH

if [ ! -d ".env" ]; then
	virtualenv -p python3 .env
fi
. .env/bin/activate

# upgrade pip
pip install --upgrade pip
# install common libraries
pip install tensorflow scipy # tensorflow includes numpy
pip install matplotlib seaborn imageio
pip install sortedcontainers
# install gym
pip install gym[atari]
# install rogue
pip install pyte
bash ./Rogue/build_with_no_monsters.sh
bash ./Rogue/build_with_monsters.sh