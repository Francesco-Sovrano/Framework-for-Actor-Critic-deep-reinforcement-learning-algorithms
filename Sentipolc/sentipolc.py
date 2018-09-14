# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import _pickle as pickle # CPickle
from datetime import datetime
import os

import flags
from build import SentiPolC

def get_set(path):
	with open(path, 'rb') as f:
		pkl = pickle.load(f)
		return pkl['training_set'], pkl['test_set']

# Build training and test set
if not os.path.isfile(flags.preprocessed_dict+'.pkl'):
	SentiPolC().build_set()
training_set, test_set = get_set(flags.preprocessed_dict+'.pkl')
# Shuffle training set (no need to shuffle test set)
np.random.shuffle(training_set)
					
def evaluate(result_file):
	print('Start evaluating')
	verbose=True
	print(datetime.now())
	# read gold standard and populate the count matrix
	gold = dict()
	gold_counts = {
					'subj':{'0':0,'1':0},
					'opos':{'0':0,'1':0},
					'oneg':{'0':0,'1':0},
					'iro':{'0':0,'1':0},
					'lpos':{'0':0,'1':0},
					'lneg':{'0':0,'1':0}
				}
	with open(flags.test_set_path) as f:
		for line in f:
			raw = line.rstrip().split(',')
			id = str(raw[0].replace('"', ''))
			subj = str(raw[1].replace('"', ''))
			opos = str(raw[2].replace('"', ''))
			oneg = str(raw[3].replace('"', ''))
			iro = str(raw[4].replace('"', ''))
			lpos = str(raw[5].replace('"', ''))
			lneg = str(raw[6].replace('"', ''))
			top = str(raw[7].replace('"', ''))
			
			#id, subj, opos, oneg, iro, lpos, lneg, top = map(lambda x: x[1:-1], line.rstrip().split(','))
			gold[id] = {'subj':subj, 'opos':opos, 'oneg':oneg, 'iro':iro, 'lpos':lpos, 'lneg':lneg}
			gold_counts['subj'][subj]+=1
			gold_counts['opos'][opos]+=1
			gold_counts['oneg'][oneg]+=1
			gold_counts['iro'][iro]+=1

			gold_counts['lpos'][lpos]+=1
			gold_counts['lneg'][lneg]+=1
				
	# read result data
	result = dict()
	with open(result_file) as f:
		for line in f:
			raw = line.rstrip().split(',')
			id = str(raw[0].replace('"', ''))
			subj = str(raw[1].replace('"', ''))
			opos = str(raw[2].replace('"', ''))
			oneg = str(raw[3].replace('"', ''))
			iro = str(raw[4].replace('"', ''))
			lpos = str(raw[5].replace('"', ''))
			lneg = str(raw[6].replace('"', ''))
			top = str(raw[7].replace('"', ''))
			result[id]= {'subj':subj, 'opos':opos, 'oneg':oneg, 'iro':iro}
			
	task_f1 = {}
	# evaluation: single classes
	for task in ['subj', 'opos', 'oneg', 'iro']:	#add 'lpos' and 'lneg' if you want to measure literal polairty
		# table header
		if verbose: print ("\ntask: {}".format(task))
		if verbose: print ("prec. 0\trec. 0\tF-sc. 0\tprec. 1\trec. 1\tF-sc. 1\tF-sc.")
		correct =  {'0':0,'1':0}
		assigned = {'0':0,'1':0}
		precision ={'0':0.0,'1':0.0}
		recall =   {'0':0.0,'1':0.0}
		fscore =   {'0':0.0,'1':0.0}
		   
		# count the labels
		for id, gold_labels in gold.items():
			if (not id in result) or result[id][task]=='':
				pass
			else:
				assigned[result[id][task]] += 1					
				if gold_labels[task]==result[id][task]:
					correct[result[id][task]] += 1

		# compute precision, recall and F-score
		for label in ['0','1']:
			try:
				precision[label] = float(correct[label])/float(assigned[label])
				recall[label] = float(correct[label])/float(gold_counts[task][label])
				fscore[label] = (2.0 * precision[label] * recall[label]) / (precision[label] + recall[label])
			except:
				# if a team doesn't participate in a task it gets default 0 F-score
				fscore[label] = 0.0
				
		task_f1[task] = (fscore['0'] + fscore['1'])/2.0
		# write down the table
		print("{0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t{4:.4f}\t{5:.4f}\t{6:.4f}".format( 
				precision['0'], recall['0'], fscore['0'], 
				precision['1'], recall['1'], fscore['1'],
				task_f1[task]))
										
	# polarity evaluation needs a further step
	if verbose: print("\ntask: polarity")
	if verbose: print("Combined F-score")
	correct =  {'opos':{'0':0,'1':0}, 'oneg':{'0':0,'1':0}}
	assigned = {'opos':{'0':0,'1':0}, 'oneg':{'0':0,'1':0}}
	precision ={'opos':{'0':0.0,'1':0.0}, 'oneg':{'0':0.0,'1':0.0}}
	recall =   {'opos':{'0':0.0,'1':0.0}, 'oneg':{'0':0.0,'1':0.0}}
	fscore =   {'opos':{'0':0.0,'1':0.0}, 'oneg':{'0':0.0,'1':0.0}}

	# count the labels
	for id, gold_labels in gold.items():
		for cl in ['opos','oneg']:
			if (not id in result) or result[id][cl]=='':
				pass
			else:
				assigned[cl][result[id][cl]] += 1
				if gold_labels[cl]==result[id][cl]:
					correct[cl][result[id][cl]] += 1
				
	# compute precision, recall and F-score
	for cl in ['opos','oneg']:
		for label in ['0','1']:
			try:
				precision[cl][label] = float(correct[cl][label])/float(assigned[cl][label])
				recall[cl][label] = float(correct[cl][label])/float(gold_counts[cl][label])
				fscore[cl][label] = float(2.0 * precision[cl][label] * recall[cl][label]) / float(precision[cl][label] + recall[cl][label])
			except:
				fscore[cl][label] = 0.0

	fscore_pos = (fscore['opos']['0'] + fscore['opos']['1'] ) / 2.0
	fscore_neg = (fscore['oneg']['0'] + fscore['oneg']['1'] ) / 2.0

	# write down the table
	task_f1["polarity"] = (fscore_pos + fscore_neg)/2.0
	print("{0:.4f}".format(task_f1["polarity"]))
	print('End evaluating')
	return task_f1