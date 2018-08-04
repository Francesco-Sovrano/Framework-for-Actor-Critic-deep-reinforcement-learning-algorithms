#!/bin/bash

MY_PATH="`dirname \"$0\"`"
cd $MY_PATH

if [ ! -d ".env" ]; then
	virtualenv -p python3 .env
fi
. .env/bin/activate

# upgrade pip
pip install --upgrade pip

pip install --compile tensorflow numpy scipy sklearn
pip install --compile matplotlib seaborn imageio
# install gym
pip install --compile gym[atari]
# install rogue
pip install --compile pyte vtk
bash ./Rogue/build_with_no_monsters.sh
bash ./Rogue/build_with_monsters.sh