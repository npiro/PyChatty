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


# ------------------------------------------------------------------------------
# original contractions function
# ------------------------------------------------------------------------------

# with this we need the following bit added to the filter_corpus and
# filter_client_agent functions, to make sure  that spellchecker doesn't filter
# out the replacement words, like she_had_she_would

contractions = {v: k for k, v in get_contractions().items()}
word_dict.update(contractions)

def replace_contractions(line):
    """
    Replaces the most common contractions in English.
    :param line: [string], one line of text
    :return: [string], line with replaced contactions
    """
    contractions = {
        "aren't":  "are_not",
        "can't": "cannot",
        "couldn't": "could_not",
        "didn't": "did_not",
        "doesn't": "does_not",
        "don't": "do_not",
        "hadn't": "had_not",
        "hasn't": "has_not",
        "haven't": "have_not",
        "he'd": "he_had_he_would",
        "he'll": "he_will_he_shall",
        "he's": "he_is_he_has",
        "i'd": "i_had_i_would",
        "i'll": "i_will_i_shall",
        "i'm": "i_am",
        "i've": "i_have",
        "isn't": "is_not",
        "it's": "it_is_it_has",
        "let's": "let_us",
        "mightn't": "might_not",
        "mustn't": "must_not",
        "shan't": "shall_not",
        "she'd": "she_had_she_would",
        "she'll": "she_will_she_shall",
        "she's": "she_is_she_has",
        "shouldn't": "should_not",
        "that's": "that_is_that_has",
        "there's": "there_is_there_has",
        "they'd": "they_had_they_would",
        "they'll": "they_will_they_shall",
        "they're": "they_are",
        "they've": "they_have",
        "we'd": "we_had_we_would",
        "we're": "we_are",
        "we've": "we_have",
        "weren't": "were_not",
        "what'll": "what_will_what_shall",
        "what're": "what_are",
        "what's": "what_is_what_has_what_does",
        "what've": "what_have",
        "where's": "where_is_where_has",
        "who'd": "who_had_who_would",
        "who'll": "who_will_who_shall",
        "who's": "who_is_who_has",
        "who've": "who_have",
        "won't": "will_not",
        "wouldn't": "would_not",
        "you'd": "you_had_you_would",
        "you'll": "you_will_you_shall",
        "you're": "you_are",
        "you've": "you_have"
    }
