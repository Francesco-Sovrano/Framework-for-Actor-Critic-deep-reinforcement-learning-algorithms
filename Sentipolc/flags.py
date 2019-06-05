# -*- coding: utf-8 -*-
import os
import multiprocessing

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))

test_set_path = os.path.join(LOCAL_PATH, 'database', 'test_set_sentipolc16.csv') # test set
training_set_path = os.path.join(LOCAL_PATH, 'database', 'training_set_sentipolc16.csv') # training set
emoji_sentiment_lexicon = os.path.join(LOCAL_PATH, 'database', 'Emoji_Sentiment_Data_v1.0.csv') # emoji sentiment lexicon
preprocessed_dict = os.path.join(LOCAL_PATH, 'database', 'preprocessed') # vectorized training set
translated_lemma_tokens = os.path.join(LOCAL_PATH, 'database', 'translated_lemma_tokens') # dictionary with translated lemma tokens
lexeme_sentiment_dict = os.path.join(LOCAL_PATH, 'database', 'lexeme_sentiment_dict') # lexeme sentiment dictionary
test_annotations = os.path.join(LOCAL_PATH, 'database', 'test_annotations')
training_annotations = os.path.join(LOCAL_PATH, 'database', 'training_annotations')
tagger_path = os.path.join(LOCAL_PATH, '.env', 'treetagger') # tagger path
nltk_data = os.path.join(LOCAL_PATH, '.env', 'nltk_data') # nltk data
word2vec_path = os.path.join(LOCAL_PATH, '.env', 'word2vec', 'cc.it.300.bin') # word2vec data
parallel_size = multiprocessing.cpu_count()