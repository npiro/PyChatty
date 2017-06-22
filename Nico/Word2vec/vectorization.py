"""
Tools to vectorize words, sentences, phrases and documents.

class SentencesGenerator(filename): iterator class to run through text line by line
"""

import os.path
import gensim, logging
import codecs


"""
class SentencesGenerator(filename):

Iterator class to run through text file line by line.
The iterator returns a split list of words in the next line in the text file
Arguments:
    filename: text file name (entire path is necessary)
"""
class SentencesGenerator(object):
    def __init__(self, filename):
        self.filename = filename
 
    def __iter__(self):
        for line in codecs.open(self.filename,'r','utf-8'):
            yield line.split()