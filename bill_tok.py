#!/usr/bin/python

import argparse
import re
import random 
import cPickle as pkl
from collections import defaultdict

parser = argparse.ArgumentParser(description='Tokenize a file')
parser.add_argument('source', type=argparse.FileType('r'))
parser.add_argument('dest', type=argparse.FileType('wb'))
args = parser.parse_args()

all_words = defaultdict(int)
data = []
idx_to_cat = []
cat_to_idx = {}
next_cat_idx = 0
for line in args.source:
	parts = line.split('\t')
	cat = parts[0]
	sent = parts[1]
	sent = re.sub(r"([^A-Z])", r" \1 ", sent)
	sent = re.sub(r"  *", r" ", sent)
	sent = re.sub(r"^ ", r"", sent)
	sent = re.sub(r" $", r"", sent)
	words = sent.split(' ')[:-1]
	data += [[cat, words]]
	for w in words:
		all_words[w] += 1 
	if cat not in cat_to_idx:
		idx_to_cat += [cat]
		cat_to_idx[cat] = next_cat_idx
		next_cat_idx += 1 

word_list = []
for k,v in all_words.iteritems():
	word_list += [(v, k)]
word_list.sort(reverse=True)

idx_to_word = []
word_to_idx = {}
for i in range(len(word_list)):
	word_to_idx[word_list[i][1]] = i
	idx_to_word += [word_list[i][1]]

random.shuffle(data)
X_out = []
y_out = []
for c, s in data:
	X_out += [[word_to_idx[w] for w in s]]
	y_out += [cat_to_idx[c]]

pkl.dump((X_out, y_out), args.dest)

