Framework for Actor-Critic deep reinforcement learning algorithms
==========
  
This software supports several deep RL and HRL algorithms (A3C, A2C, PPO, GAE, etc..) in different environments (OpenAI's Gym, Rogue, Sentiment Analysis, Car Controller, etc..) with continuous and discrete action spaces. 
Actor-Critic is a big family of RL algorithms. In this work we want to focus primarily on:
* Actor-Critic paradigm
* Hierarchical networks
* Experience Replay
* Exploration intrinsic rewards

In order to do so, we need a framework for experimenting RL algorithms, with ease, on different types of problem.
In May 2017, [OpenAI](https://github.com/openai/baselines) has published an interesting repository of RL baselines, and it is still maintaining it with continuous improvements and updates.
The aforementioned repository is probably the best choice for testing the performances of already existing RL algorithms requiring very minimal changes, but it is hard to read and modify (at least for the author).
Thus we decided to use as code-base for our experiments the open-source A3C algorithm, built on Tensorflow 1.10.1, that comes with our last conference paper [Crawling in Rogue's dungeons with (partitioned) A3C](https://arxiv.org/abs/1804.08685v1), mainly because we already had experience with it and we know the details of its inner mechanisms. But even the chosen code-base is not generic and abstract enough for our goals, for example that code-base is made for Rogue only, thus we had to make some changes to it:
* We created a unique configuration file in the Framework root directory, for configuring and combining with ease (in a single point) all the Framework features, algorithms, methods, environments, etc.. (included those that are going to be mentioned in the following points of this enumeration)
* Added support for all the Atari games available in the [OpenAI Gym repository](https://github.com/openai/baselines).
* Created a new environment for Sentiment Analysis.
* Created a new environment for Car Controller.
* Added support for A2C.
* Added Experience Replay and Prioritized Experience Replay.
* Added Count-Based Exploration.
* Added PPO and PVO.
* In many OpenAI baselines the vanilla policy and value gradient has been slightly modified in order to perform a reduce mean instead of a reduce sum, because this way it is possible to reduce numerical errors when training with huge batches. Thus, it has been added support for both mean-based and sum-based losses.
* Added GAE.
* Added support for all the gradient optimizers supported by Tensorflow 1.10.1: Adam, Adagrad, RMSProp, ProximalAdagrad, etc..
* Added support for global gradient norm clipping and learning rate decay using some of the decay functions supported by Tensorflow 1.10.1: exponential decay, inverse time decay, natural exp decay.
* Added different generic hierarchical structures based on the Options Framework for partitioning the state space using:
	* K-Means clustering
	* Reinforcement Learning
* Made possible to create and use new neural network architectures, simply extending the base one. The base neural network, by default, allows to share network layers between the elements of the hierarchy: parent, siblings.
* In order to simplify experiments analysis, it is required a mechanism for an intuitive graphic visualization. We implemented an automatic system for generating GIFs of all episodes observations, and an automatic plotting system for showing statistics of training and testing. The plotting system can also be used to easily compare different experiments (every experiment is colored differently in the plot).
* For Rogue environment, we implemented a debugging mechanism that allows to visualize (also inside episode GIFs) the heatmap of the value function of an agent.
* Added variations of the auxiliary techniques described in [Reinforcement Learning with Unsupervised Auxiliary Tasks](https://arxiv.org/abs/1611.05397): Reward Prediction.
* Added support for continuous control.
* Added support for multi-action control.

The software used for Rogue environment is a fork of:
* [RogueInABox](https://github.com/rogueinabox/rogueinabox)
* [Partitioned A3C for RogueInABox](https://github.com/Francesco-Sovrano/Partitioned-A3C-for-RogueInABox)

The software used for Sentiment Analysis environment is a fork of:
* [Generic Hierarchical Deep Reinforcement Learning for Sentiment Analysis](https://github.com/Francesco-Sovrano/Generic-Hierarchical-Deep-Reinforcement-Learning-for-Sentiment-Analysis)

This project has been tested on Debian 9. The setup.sh script installs the necessary dependencies.
Dependencies shared by all environments:
* [Tensorflow](https://www.tensorflow.org/)
* [Scipy](https://www.scipy.org/)
* [Scikit learn](http://scikit-learn.org/stable/index.html)
* [Matplotlib](https://matplotlib.org/)
* [Seaborn](https://seaborn.pydata.org/)
* [ImageIO](https://imageio.github.io/)
* [Sorted Containers](https://pypi.org/project/sortedcontainers/)

Video-games environment dependencies:
* [Gym](https://gym.openai.com/)
* [Pyte](https://pypi.org/project/pyte/)

Sentiment Analysis environment dependencies:
* [TreeTagger](http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/)
* [fastText](https://fasttext.cc/docs/en/crawl-vectors.html) pre-trained model for Italian
* [VU sentiment lexicon](https://github.com/opener-project/VU-sentiment-lexicon) by [OpeNER](http://www.opener-project.eu/)
* [emojipy](https://github.com/launchyard/emojipy) a Python library for working with emoji
* [NLTK](http://www.nltk.org/)
* [Googletrans](https://pypi.org/project/googletrans/2.2.0/), a free and unlimited python library that implemented Google Translate API.
* [gensim](https://radimrehurek.com/gensim/)

Before running setup.sh you must have installed virtualenv, python3-dev, python3-pip and make. 
For more details, please read the related paper.

The train.sh script starts the training.
The test.sh script evaluates the trained agent using the weights in the most recent checkpoint.