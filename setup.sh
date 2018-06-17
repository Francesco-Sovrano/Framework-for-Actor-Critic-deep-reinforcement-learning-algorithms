#!/bin/bash

MY_PATH="`dirname \"$0\"`"
cd $MY_PATH

if [ ! -d ".env" ]; then
	virtualenv -p python3 .env
fi
. .env/bin/activate
pip install --upgrade pip

# install OpenAI baselines
cd .env
if [ ! -d "baselines" ]; then
	git clone https://github.com/openai/baselines.git
fi
cd ./baselines
pip install -e .
cd ../..

pip install tensorflow matplotlib numpy scipy scikit-image pyte vtk sklearn
pip install gym[all]
pip install pytest
pytest

bash ./Rogue/build_with_no_monsters.sh
bash ./Rogue/build_with_monsters.sh