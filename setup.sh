#!/bin/bash

MY_DIR="`dirname \"$0\"`"
MY_PATH="`realpath $MY_DIR`"
cd $MY_PATH

if [ ! -d ".env" ]; then
	virtualenv -p python3 .env
fi
. .env/bin/activate

# upgrade pip
pip install pip==9.0.3 # pip 10.0.1 has issues with pybind11 -> required by fastText
# install common libraries
pip install tensorflow==1.10.1 scipy sklearn # tensorflow includes numpy
pip install matplotlib seaborn imageio
pip install sortedcontainers
# install gym
pip install gym[atari]
# install rogue
pip install pyte
$MY_PATH/Rogue/build_with_no_monsters.sh
$MY_PATH/Rogue/build_with_monsters.sh
# install sentipolc
pip install googletrans==2.2.0 gensim==3.4.0 validate_email==1.3
pip install nltk==3.2.5 treetaggerwrapper==2.2.4 git+https://github.com/facebookresearch/fastText.git@3e64bf0f5b916532b34be6706c161d7d0a4957a4 # the Moses tokenizer has been removed from nltk 3.3.0!
pip install emojipy==3.0.5 # https://github.com/emojione/emojione/tree/master/lib/python
# pip install lxml git+https://github.com/opener-project/VU-sentiment-lexicon.git # this version of VU-sentiment-lexicon is for python2 only
pip install lxml==4.2.1 git+https://github.com/Francesco-Sovrano/VU-sentiment-lexicon.git
cd ./.env
# install treetagger
if [ ! -d "treetagger" ]; then
	mkdir treetagger
	cd ./treetagger
	wget http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/tree-tagger-linux-3.2.1.tar.gz
	wget http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/tagger-scripts.tar.gz
	wget http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/install-tagger.sh
	wget http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/italian.par.gz
	wget http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/english.par.gz
	chmod -R 700 ./
	./install-tagger.sh
	if [ -r italian.par.gz ]
	then
		gzip -cd italian.par.gz > lib/italian-utf8.par
		echo 'Italian parameter file re-installed.'
	fi
	if [ -r english.par.gz ]
	then
		gzip -cd english.par.gz > lib/english-utf8.par
		echo 'English parameter file re-installed.'
	fi
	cd ..
fi
# Download pre-trained word vectors from fasttext repository
if [ ! -d "word2vec" ]; then
	mkdir word2vec
	cd ./word2vec
	wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/word-vectors-v2/cc.it.300.bin.gz
	gunzip cc.it.300.bin.gz
	cd ..
fi
cd ..
# Build preprocessed vectors
python3 $MY_PATH/Sentipolc/build.py