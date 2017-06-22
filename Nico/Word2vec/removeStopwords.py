# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 15:37:36 2016

@author: daniel
"""

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

# load stopwords
stop = set(stopwords.words('english'))
# regexp for removing punctuation
tokenizer = RegexpTokenizer(r'\w+')

# open files
clientMessages = open('clientMessages.txt')
clientMessagesFiltered = open('clientMessagesFiltered.txt', 'w')

def join_line(line_list):
    """
    Joins a list of strings into one string.
    """
    return ' '.join(line_list)
    

for line in clientMessages:
    no_punctuation = join_line(tokenizer.tokenize(line))
    no_stopwords = [w for w in no_punctuation.lower().split() if w not in stop]
    clientMessagesFiltered.write(join_line(no_stopwords) + '\n')