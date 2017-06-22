# ------------------------------------------------------------------------------
# Getting Indian and other names from text
# ------------------------------------------------------------------------------
import re, codecs, os
data_folder = '/home/daniel/s2ds/Data/'
input_file = '04_doc2vecTrainingDataFiltered2.txt'
output_file = 'otherNames.txt'
input_file = codecs.open(os.path.join(data_folder, input_file), 'rU', 'utf-8')
output_file = codecs.open(os.path.join(data_folder, output_file), 'w', 'utf-8')
names_dict = {}
for i, line in enumerate(input_file):
    name_match = re.match(r"^(hi this is )(\w+) ", line)
    if name_match:
        name = name_match.group(2)
        if name not in ['param_female_name', 'param_male_name'] and name not in names_dict:
            names_dict[name] = 1
            output_file.write(name + '\n')
input_file.close()
output_file.close()


# ------------------------------------------------------------------------------
# Make female and male names files un-redundant and join them with the otherNames
# ------------------------------------------------------------------------------

import re, codecs, os
data_folder = '/home/daniel/s2ds/Data/'
input_files = ['femaleNames.txt', 'maleNames.txt', 'otherNames.txt']
output_file = 'names.txt'
output_file = codecs.open(os.path.join(data_folder, output_file), 'w', 'utf-8')
names_dict = {}
for f in input_files:
    input_file = codecs.open(os.path.join(data_folder, f), 'rU', 'utf-8')
    for i, line in enumerate(input_file):
        name = line.strip()
        if name not in names_dict:
            names_dict[name] = 1
            output_file.write(name + '\n')
    input_file.close()
output_file.close()


# ------------------------------------------------------------------------------
# Getting nouns for Walter
# ------------------------------------------------------------------------------


import os, nltk, codecs
data_folder = '/home/daniel/s2ds/Data/'
input_file = '03_clientMessages.txt'
output_file = '03_clientMessagesOnlyNouns.txt'
input_file = codecs.open(os.path.join(data_folder, input_file), 'rU', 'utf-8')
output_file = codecs.open(os.path.join(data_folder, output_file), 'w', 'utf-8')

for i, line in enumerate(input_file):
    if i < 10:
        new_line = ''
        for word_type in nltk.pos_tag(nltk.word_tokenize(line)):
            if word_type[1] in ['NN', 'NNP', 'NNPS', 'NNS']:
                new_line += ' ' + word_type[0].lower()
        output_file.write(new_line + '\n')
input_file.close()
output_file.close()

# ------------------------------------------------------------------------------
# Parallel NLTKPreprocessor
# Tried to speed up the lemmatizer by calling it in parallel on all words in a
# line. It uses a lot more CPUs but it's not faster (indeed a lot slower) than
# using the word_cache trick, so I abandoned the idea.
# ------------------------------------------------------------------------------

import re
import os
import codecs
import string
import numpy as np
from itertools import izip
from joblib import Parallel, delayed
import gensim as gs
from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk import sent_tokenize
from nltk import pos_tag
from sklearn.base import BaseEstimator, TransformerMixin


def tokenize_parallel(token, tag, lower, strip, stopwords, punct, word_cache, lemmatizer):


    token = token.lower() if lower else token
    token = token.strip() if strip else token
    token = token.strip('_') if strip else token
    token = token.strip('*') if strip else token

    # If stopword, ignore token and continue
    if token in stopwords:
        return ''

    # If punctuation, ignore token and continue
    if all(char in punct for char in token):
        return ''

    # Lemmatize the token and yield
    if token not in word_cache:
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(tag[0], wn.NOUN)
        lemma = lemmatizer.lemmatize(token, tag)
        word_cache[lemma] = 1
        return lemma
    else:
        return token


class NLTKPreprocessorParallel(BaseEstimator, TransformerMixin):
    """
    Scikit-learn like class with fit, transform and fit_transform methods, which
    removes stopwords, punctuation, plus makes everything lower case and
    lemmatizes the text.
    """
    def __init__(self, stopwords=None, punct=None, lower=True, strip=True):
        self.lower = lower
        self.strip = strip
        self.stopwords = stopwords or set(sw.words('english'))
        self.punct = punct or set(string.punctuation)
        self.lemmatizer = WordNetLemmatizer()
        self.word_cache = {}

    def fit(self, document, y=None):
        return self

    def transform(self, document):
        return self.tokenize(document)



    def tokenize(self, document):
        # Break the document into sentences


        for sent in sent_tokenize(document):
            tokens = np.array(Parallel(n_jobs=-1)
                              (delayed(tokenize_parallel)(token, tag,
                              self.lower, self.strip, self.stopwords,
                              self.punct, self.word_cache, self.lemmatizer)
                              for token, tag in pos_tag(wordpunct_tokenize(sent))))

            return tokens

