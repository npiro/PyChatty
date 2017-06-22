# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 19:10:40 2016

@author: Client
"""

import preprocessing.filtering as filtering
# from preprocessing.filtering import NLTKPreprocessor
from nltk.corpus import words

class InputCleaner(object):

    def __init__(self, data_folder = '.'):
        # list of words for spellcheck
        self.word_dict = {w: 1 for w in words.words()}
        self.names = filtering.build_names(data_folder)
        
    def clean_input(self, sentence):
        """
        Apply all preprocessing to input line
        """
        # nltk_preproc = NLTKPreprocessor()

        sent_clean = filtering.filter_line(sentence, preprocessing=True, names=self.names, spellcheck=self.word_dict)
        
        return sent_clean