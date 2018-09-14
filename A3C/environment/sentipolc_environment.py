# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from environment import environment
import numpy as np
import os
import math
from collections import deque
import copy

import options
flags = options.get()
import sys
sys.path.append(flags.sentipolc_path)
import sentipolc

class SentiPolcEnvironment(environment.Environment):
	
	def __init__(self, thread_index, training):
		self.gram_size = 1 # number of tokens/lemma to process at each step
		self.granularity = "lemma" # "lemma" or "token"
		self.task = "subjective, opos, oneg, ironic, lpos, lneg" # choose a combination of: subjective, opos, oneg, ironic, lpos, lneg
		# get global documents, shuffle
		if training:
			self.shuffle = True
			self.documents = sentipolc.training_set # passage by reference
		else:
			self.shuffle = False # no need to shuffle test set
			self.documents = sentipolc.test_set # passage by reference
		self.task_id = [task.strip() for task in self.task.split(',')]
		self.task_count = len(self.task_id)
		self.reward_list = deque()
		self.confusion_list = deque()
		self.thread_index = thread_index
		self.epoch = 0
		self.id = self.thread_index
		
	def get_action_shape(self):
		return (self.task_count,2) # take n actions of 2 possible types

	def get_concatenation_size(self):
		return self.task_count
		
	def get_concatenation(self):
		return self.sentidoc
		
	def get_state_shape(self):
		return ( self.gram_size, 300+6, 1 )

	def get_state(self):
		return self.last_state
	
	def compute_state(self):
		state = np.zeros( self.get_state_shape() )
		for i in range(self.gram_size):
			word_index = self.step + i
			if word_index < self.max_step-1:
				token = self.tokens[word_index]
				v = np.concatenate( (token[self.granularity+"_vector"]*100,self.get_word_meta(token)) )
				state[i] = np.expand_dims(v, axis=-1)
			elif word_index == self.max_step-1:
				v = np.concatenate( (self.docvec*100,self.get_word_meta(None)) )
				state[i] = np.expand_dims(v, axis=-1)
		return state
		
	def get_word_meta(self, token):
		if token is None:
			return [0, 0, 0, 0, 0, 16] # is docvec
		lexeme_sentiment = token["lexeme_sentiment"]
		negativity = 0
		positivity = 0
		is_emoji = 0
		if len(lexeme_sentiment)>0:
			negativity = lexeme_sentiment["0"]["negativity"]
			positivity = lexeme_sentiment["0"]["positivity"]
			if "shortcode" in lexeme_sentiment["0"]:
				is_emoji = 1
		return [negativity, positivity, token["is_negator"]*2, token["is_intensifier"]*4, is_emoji*8, 0]
				
	def compute_reward(self):
		reward = 0
		confusion = self.get_confusion_matrix()
		for i in range(self.task_count):
			key = self.get_task_by_index(i)
			tp = confusion[key][0][0]
			fp = confusion[key][0][1]
			tn = confusion[key][1][1]
			fn = confusion[key][1][0]
			if not self.terminal: # doesn't work without it
				if tp:
					reward += 0.05
				elif fn:
					reward += -0.05
			else:
				if tp:
					reward += 1
				elif tn:
					reward += 0.1 # doesn't work without it
				elif fp:
					reward += -1
				elif fn:
					reward += -0.1
		return reward
		
	def reset(self):
		self.id += flags.parallel_size
		if self.id >= len(self.documents): # start new epoch
			if self.shuffle and self.thread_index==0:
				np.random.shuffle(self.documents)
			self.id = self.thread_index
			self.epoch += 1
		# retrieve annotated wordbag
		doc = self.documents[self.id]
		self.tokens = self.remove_noisy_tokens( doc["tokens_annotation"] ) # no stopwords
		self.annotation = doc["text_annotation"]
		self.docvec = doc["average_docvec"]
		# init episode
		self.terminal = False
		self.episode_reward = 0
		self.step = 0
		self.max_step = len(self.tokens)+1 # words + context (docvec)
		# get state
		self.sentidoc = np.zeros(self.task_count, dtype=np.uint8)
		self.last_state = self.compute_state()
		
	def remove_noisy_tokens(self, tokens):
		new_tokens = []
		for token in tokens:
			if token["is_stop"]==0 and token["is_uri"]==0:
				new_tokens.append(token)
		return new_tokens
		
	def choose_action(self, action_vector):
		action = []
		shape = np.shape(action_vector)
		for i in range(shape[0]):
			action.append(np.argwhere(action_vector[i]==1)[0][0])
		return action
		
	def process(self, action_vector):
		action = self.choose_action(action_vector)
		if self.step+self.gram_size >= self.max_step:
			self.terminal = True # do it before computing reward
		self.old_prediction = self.sentidoc
		self.sentidoc = action
		
		if self.terminal: # Confusion
			self.confusion_list.append( self.get_confusion_matrix() )
			if len(self.confusion_list) > flags.match_count_for_evaluation:
				self.confusion_list.popleft()
		
		reward = self.compute_reward()
		self.episode_reward += reward
		
		if self.terminal: # Reward
			self.reward_list.append(self.episode_reward)
			if len(self.reward_list) > flags.match_count_for_evaluation:
				self.reward_list.popleft()
		
		self.step += self.gram_size # it changes the current state, do it as last command of this function (otherwise error!!)
		self.last_state = self.compute_state() # compute new state after updating step
		return self.last_state, reward, self.terminal
		
	def stop(self):
		pass
	
	def get_task_by_index(self, index):
		return self.task_id[index]
	
	def get_labeled_prediction(self):
		dict = {}
		for i in range(self.task_count):
			key = self.get_task_by_index(i)
			value = self.sentidoc[i]
			dict[key] = value
		return dict
		
	def get_confusion_matrix(self):
		confusion = {}
		for i in range(self.task_count):
			key = self.get_task_by_index(i)
			value = self.sentidoc[i]
			confusion[key] = np.zeros((2,2))
			# Win
			if self.annotation[key] == value:
				if value == 1: # true positive
					confusion[key][0][0] = 1
				else: # true negative
					confusion[key][1][1] = 1
			# Lose
			else:
				if value == 1: # false positive
					confusion[key][0][1] = 1
				else: # false negative
					confusion[key][1][0] = 1
		return confusion

	def get_frame_info(self, network, value, action, reward, policy):
		state_info = "reward={}, agent={}, value={}, policy={}\n".format(reward, network.agent_id, value, policy)
		action_info = "action={}\n".format(action)
		observation_info = "observation={}\n".format(self.last_state)
		concat_info = "concat={}\n".format(self.sentidoc)
		frame_info = { "log": state_info + action_info + observation_info + concat_info }
		return frame_info
	
	def get_statistics(self):
		tp = {} # true positive
		tn = {} # true negative
		fp = {} # false positive
		fn = {} # false negative
		for i in range(self.task_count):
			key = self.get_task_by_index(i)
			tp[key]=0
			tn[key]=0
			fp[key]=0
			fn[key]=0
		for confusion in self.confusion_list:
			for key, value in confusion.items():
				tp[key] += value[0][0]
				tn[key] += value[1][1]
				fp[key] += value[0][1]
				fn[key] += value[1][0]
		stats = {}
		stats["avg_reward"] = sum(self.reward_list)/len(self.reward_list) if len(self.reward_list) != 0 else 0
		stats["epoch"] = self.epoch
		for i in range(self.task_count):
			key = self.get_task_by_index(i)
			# stats[key+"_p+"], stats[key+"_r+"], stats[key+"_f1+"] = self.get_positive_fscore(tp[key], tn[key], fp[key], fn[key])
			# stats[key+"_p-"], stats[key+"_r-"], stats[key+"_f1-"] = self.get_negative_fscore(tp[key], tn[key], fp[key], fn[key])
			_, _, stats[key+"_f1+"] = self.get_positive_fscore(tp[key], tn[key], fp[key], fn[key])
			_, _, stats[key+"_f1-"] = self.get_negative_fscore(tp[key], tn[key], fp[key], fn[key])
			# stats[key+"_precision"] = (stats[key+"_p+"]+stats[key+"_p-"])/2
			# stats[key+"_recall"] = (stats[key+"_r+"]+stats[key+"_r-"])/2
			stats[key+"_f1"] = (stats[key+"_f1+"]+stats[key+"_f1-"])/2
			# stats[key+"_accuracy"] = self.get_accuracy(tp[key], tn[key], fp[key], fn[key])
			stats[key+"_mcc"] = self.get_mcc(tp[key], tn[key], fp[key], fn[key])
		stats["avg_mcc"] = sum(stats[self.get_task_by_index(i)+"_mcc"] for i in range(self.task_count))/self.task_count
		return stats
		
	def get_positive_fscore(self, tp, tn, fp, fn):
		precision = tp/(tp+fp) if tp+fp != 0 else 0
		recall = tp/(tp+fn) if tp+fn != 0 else 0
		f1 = 2 * ((precision*recall)/(precision+recall)) if precision+recall != 0 else 0
		return precision, recall, f1
		
	def get_negative_fscore(self, tp, tn, fp, fn):
		precision = tn/(tn+fn) if tn+fn != 0 else 0
		recall = tn/(tn+fp) if tn+fp != 0 else 0
		f1 = 2 * ((precision*recall)/(precision+recall)) if precision+recall != 0 else 0
		return precision, recall, f1
		
	def get_mcc(self, tp, tn, fp, fn): # https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
		denominator = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
		if denominator != 0:
			return (tp*tn - fp*fn)/denominator
		return 2*self.get_accuracy(tp, tn, fp, fn)-1
	
	def get_accuracy(self, tp, tn, fp, fn):
		accuracy = (tp+tn)/(tp+tn+fp+fn) if tp+tn+fp+fn != 0 else 0
		return accuracy
		
	def get_test_result(self):
		annotation = copy.deepcopy(self.annotation)
		for i in range(self.task_count):
			key = self.get_task_by_index(i)
			annotation[key] = self.sentidoc[i]
		return '{0},{1},{2},{3},{4},{5},{6},{7}\n'.format( annotation["id"], annotation["subjective"], annotation["opos"], annotation["oneg"], annotation["ironic"], annotation["lpos"], annotation["lneg"], annotation["topic"] )
		
	def get_test_size(self):
		return len(self.documents)//flags.parallel_size
		
	def evaluate_test_results(self, test_result_file):
		return sentipolc.evaluate(test_result_file)