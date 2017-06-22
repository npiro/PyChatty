
"""
This submodule has methods to filter stopwords, punctuation, and lemmatize the
input text, i.e. turning bunny, bunnies, Bunny, bunny!, and _bunny_ all into a
single feature bunny.

Furthermore it replaces Britsh postcodes, any form of numbers (telephone, order,
purchase, etc) and English names into predefined strings.
"""

import re
import os
import codecs
import pandas as pd
import numpy as np
from itertools import izip
import gensim as gs
from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk.corpus import words
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk import sent_tokenize
from nltk import pos_tag
from sklearn.base import BaseEstimator, TransformerMixin


def filter_corpus(data_folder, input_file, output_file, preprocessing=False,
                  names=None, spellcheck=True, ngram=None, lemmatize=False,
                  nltk_preproc=None):
    """
    Filters a text file using a set of filters defined by the boolean variables.

    :param data_folder: [string], absolute path to data folder
    :param input_file:  [string], name of input corpus file
    :param output_file: [string], name of the output file
    :param preprocessing: [boolean], whether to run a series of preproc steps
    :param names: [dict], if preprocessing is True, this is a dict with names
    :param spellcheck: [boolean], whether to run simple spellcheck
    :param ngram: [obj], if not None, holds a ngram model from get_ngram_model
    :param lemmatize [boolean], weather to lemmatize corpus, this  is slow
    :param nltk_preproc [obj], with this we can pass in a lemmatized corpus dict
    :return: if lemmatize, return the nltk_preproc with "trained" word-cache dict
    """
    # open files for reading and writing
    input_file = codecs.open(os.path.join(data_folder, input_file), 'rU', 'utf-8')
    output_file = codecs.open(os.path.join(data_folder, output_file), 'w', 'utf-8')

    # init preprocessor if needed, otherwise use provided, or none at all
    if lemmatize and nltk_preproc is None:
        nltk_preproc = NLTKPreprocessor()

    if spellcheck:
        word_dict = {w: 1 for w in words.words()}
    else:
        word_dict = None

    for i, line in enumerate(input_file):
        line = line.strip()
        line = filter_line(line, preprocessing, names, word_dict, ngram,
                           nltk_preproc)
        # if no or only one word left, ignore line
        if len(line.split()) > 1:
            output_file.write(line + '\n')

    # close files
    input_file.close()
    output_file.close()

    if lemmatize:
        return nltk_preproc


def filter_client_agent_conv(data_folder, client_file, agent_file,
                             client_output, agent_output, conv_output,
                             preprocessing=False, names=None, spellcheck=False,
                             ngram=None, lemmatize=False, nltk_preproc=None):
    """
    Filters the paired client and agent files together making sure they match.
    It also appends the client_agent_summary.csv to keep track of the sentences
    that were removed from the corpus.

    This version also creates a file with one conversation per line and saves
    the indexing of it to the summary file.

    :param data_folder: [string], absolute path to data folder
    :param client_file:  [string], name of client file
    :param agent_file: [string], name of agent file
    :param client_output:  [string], name of filtered client file
    :param agent_output: [string], name of filtered agent file
    :param conv_output: [string], name of file for conversations
    :param preprocessing: [boolean], whether to run a series of preproc steps
    :param names: [dict], if preprocessing is True, this is a dict with names
    :param spellcheck: [obj], whether to run simple spellcheck
    :param ngram: [obj], if not None, holds a ngram model from get_ngram_model
    :param lemmatize [boolean], weather to lemmatize corpus, this  is slow
    :param nltk_preproc [obj], with this we can pass in a lemmatized corpus dict
    :return: None
    """
    # open files for reading and writing
    client_file = codecs.open(os.path.join(data_folder, client_file), 'rU', 'utf-8')
    client_output = codecs.open(os.path.join(data_folder, client_output), 'w', 'utf-8')
    agent_file = codecs.open(os.path.join(data_folder, agent_file), 'rU', 'utf-8')
    agent_output = codecs.open(os.path.join(data_folder, agent_output), 'w', 'utf-8')
    conv_output = codecs.open(os.path.join(data_folder, conv_output), 'w', 'utf-8')
    summary_file = os.path.join(data_folder, 'client_agent_summary.csv')
    summary_file2 = os.path.join(data_folder, 'client_agent_summary3.csv')
    summary = pd.read_csv(summary_file, index_col=0)
    summary.insert(4, 'linePosFiltered', range(summary.shape[0]))
    summary.insert(5, 'convIDFiltered', range(summary.shape[0]))

    # init preprocessor if needed, otherwise use provided, or none at all
    if lemmatize and nltk_preproc is None:
        nltk_preproc = NLTKPreprocessor()

    if spellcheck:
        word_dict = {w: 1 for w in words.words()}
    else:
        word_dict = None

    # flags to keep track of the filtered sentences/conversations
    line_count = 0
    new_line_count = 0
    old_conv_count = 0
    new_conv_count = 0
    conv = []
    with client_file as client, agent_file as agent:
        for client_line, agent_line in izip(client, agent):
            client_line = client_line.strip()
            agent_line = agent_line.strip()
            client_line = filter_line(client_line, preprocessing, names,
                                      word_dict, ngram, nltk_preproc)
            agent_line = filter_line(agent_line, preprocessing, names,
                                     word_dict, ngram, nltk_preproc)
            if summary.iloc[line_count, 0] != old_conv_count:
                old_conv_count = summary.iloc[line_count, 0]
                if len(conv) > 0:
                    conv_output.write(' '.join(conv) + '\n')
                    conv = []
                    new_conv_count += 1

            if len(client_line.split()) > 0 and len(agent_line.split()) > 0:
                client_output.write(client_line + '\n')
                conv.append(client_line)
                agent_output.write(agent_line + '\n')
                conv.append(agent_line)
                summary.iloc[line_count, 4] = new_line_count
                summary.iloc[line_count, 5] = new_conv_count
                new_line_count += 1
            else:
                summary.iloc[line_count, 4] = np.nan
                summary.iloc[line_count, 5] = np.nan
            line_count += 1

        # otherwise we miss the last conversation
        if len(conv) > 0:
            conv_output.write(' '.join(conv) + '\n')

    # close files
    client_file.close()
    client_output.close()
    agent_file.close()
    agent_output.close()
    conv_output.close()
    summary.to_csv(summary_file2)


def filter_client_agent(data_folder, client_file, agent_file, client_output,
                        agent_output, preprocessing=False, names=None,
                        spellcheck=False, ngram=None, lemmatize=False,
                        nltk_preproc=None):
    """
    Filters the paired client and agent files together making sure they match.
    It also appends the client_agent_summary.csv to keep track of the sentences
    that were removed from the corpus.

    :param data_folder: [string], absolute path to data folder
    :param client_file:  [string], name of client file
    :param agent_file: [string], name of agent file
    :param client_output:  [string], name of filtered client file
    :param agent_output: [string], name of filtered agent file
    :param preprocessing: [boolean], whether to run a series of preproc steps
    :param names: [dict], if preprocessing is True, this is a dict with names
    :param spellcheck: [obj], whether to run simple spellcheck
    :param ngram: [obj], if not None, holds a ngram model from get_ngram_model
    :param lemmatize [boolean], weather to lemmatize corpus, this  is slow
    :param nltk_preproc [obj], with this we can pass in a lemmatized corpus dict
    :return: None
    """
    # open files for reading and writing
    client_file = codecs.open(os.path.join(data_folder, client_file), 'rU', 'utf-8')
    client_output = codecs.open(os.path.join(data_folder, client_output), 'w', 'utf-8')
    agent_file = codecs.open(os.path.join(data_folder, agent_file), 'rU', 'utf-8')
    agent_output = codecs.open(os.path.join(data_folder, agent_output), 'w', 'utf-8')
    summary_file = os.path.join(data_folder, 'client_agent_summary.csv')
    summary_file2 = os.path.join(data_folder, 'client_agent_summary2.csv')
    summary = pd.read_csv(summary_file, index_col=0)
    summary.insert(4, 'linePosFiltered', range(summary.shape[0]))

    # init preprocessor if needed, otherwise use provided, or none at all
    if lemmatize and nltk_preproc is None:
        nltk_preproc = NLTKPreprocessor()

    if spellcheck:
        word_dict = {w: 1 for w in words.words()}
    else:
        word_dict = None

    line_count = 0
    new_line_count = 0
    with client_file as client, agent_file as agent:
        for client_line, agent_line in izip(client, agent):
            client_line = client_line.strip()
            agent_line = agent_line.strip()
            client_line = filter_line(client_line, preprocessing, names,
                                      word_dict, ngram, nltk_preproc)
            agent_line = filter_line(agent_line, preprocessing, names,
                                     word_dict, ngram, nltk_preproc)

            if len(client_line.split()) > 0 and len(agent_line.split()) > 0:
                client_output.write(client_line + '\n')
                agent_output.write(agent_line + '\n')
                summary.iloc[line_count, 4] = new_line_count
                new_line_count += 1
            else:
                summary.iloc[line_count, 4] = np.nan
            line_count += 1

    # close files
    client_file.close()
    client_output.close()
    agent_file.close()
    agent_output.close()
    summary.to_csv(summary_file2)


def filter_line(line, preprocessing=False, names=None, spellcheck=None,
                ngram=None, lemmatize=None):
    """
    Filters a line of text using a set of filters defined by the booleans.
    The order of these filters is really important. Do NOT change it.

    :param line: [string], text to be filtered
    :param preprocessing: [boolean], if true: whole bunch of things are replaced
                          check the code: ms, numbers, names, postcodes, etc..
    :param names: [dict], if preprocessing is True, this is a dict with names
    :param spellcheck: [dict], if not None, a dict of NLTK's words list.
    :param ngram: [obj], if not None, holds a ngram model from get_ngram_model
    :param lemmatize: [obj], if not None line will be lemmatized
    :return: [string] filtered line
    """

    # replace all (hopefully) MS text
    if preprocessing:
        line = line.lower()
        line = replace_ms(line)
        line = replace_emails(line)
        line = replace_postcode(line)
        line = replace_dates(line)
        line = replace_prices(line)
        line = replace_numbers(line)
        line = replace_emoticon(line)
        # replace he's with he is, etc
        line = replace_contractions(line)
        line = join_line(remove_stopwords(line))
        line = line.replace("'", "")
        # remove punctuation except for ', _ and || separator in the doc2vec
        tokenizer = RegexpTokenizer(r"[\w]+|\|{2}")
        line = join_line(tokenizer.tokenize(line))
        line = replace_names(line, names)

    if spellcheck is not None:
        line = join_line(simple_spellcheck(line, spellcheck))

    if ngram is not None:
        line = join_line(ngram[line])

    if lemmatize is not None:
        line = join_line(lemmatize.fit_transform(line))

    return line


def filter_agent_line(line, names=None):
    """
    Filters a line of agent response, but keeping it human readable, i.e. 
    many of the preprocessing steps in filter_line() are not performed.
    
    :param line: [string], text to be filtered
    :param names: [dict], if preprocessing is True, this is a dict with names
    :return: [string] filtered line
    """

    line = replace_ms(line)
    line = replace_emails(line)
    line = replace_postcode(line)
    line = replace_dates(line)
    line = replace_prices(line)
    line = replace_numbers(line)
    line = replace_emoticon(line)
    line = replace_names(line, names)

    return line


class NLTKPreprocessor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn like class with fit, transform and fit_transform methods, which
    removes stopwords, and lemmatizes the text.
    """
    def __init__(self, stopwords=None):
        self.stopwords = stopwords or set(sw.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.word_cache = {}

    def fit(self, document, y=None):
        return self

    def transform(self, document):
        return self.tokenize(document)

    def tokenize(self, document):
        # Break the document into sentences
        for sent in sent_tokenize(document):
            # Break the sentence into part of speech tagged tokens
            for token, tag in pos_tag(wordpunct_tokenize(sent)):
                # If stopword, ignore token and continue
                if token in self.stopwords:
                    continue

                # Lemmatize the token and yield
                if token not in self.word_cache:
                    lemma = self.lemmatize(token, tag)
                    self.word_cache[token] = lemma
                    yield lemma
                else:
                    yield self.word_cache[token]

    def lemmatize(self, token, tag):
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(tag[0], wn.NOUN)
        return self.lemmatizer.lemmatize(token, tag)


def join_line(line_list, sep=' '):
    """
    Joins a list of strings into one string separated by the sep param.

    :param line_list [list], list of strings to be joined.
    :param sep [str], string to use when joining the elements of line_list
    """
    return sep.join(line_list)


def remove_stopwords(line):
    """
    Removes English stopwords that appear in the NLTK stopwords library.

    :param line: [string], one line of text
    :return: [string], where stopwords are replaced with ''
    """
    stopwords = set(sw.words('english'))
    new_line = []
    for word in wordpunct_tokenize(line):
        if word not in stopwords:
            new_line.append(word)
    return new_line


def replace_ms(line, repl=' param_ms  '):
    """
    Replaces most imaginable MS version with a single token. Also turns
    everything into lower case.

    :param line: [string], one line of text
    :param repl: [string], string to replace MS equivalent text with
    :return: [string], where MS string are replaced with repl
    """
    ms_versions = [
        "marks\s*and\s*spencer",
        "mark\s*and\s*spencers",
        "mark\s*and\s*spencer",
        "marks\s*and\s*spencers",
        "marks\s*n\s*spencer",
        "mark\s*n\s*spencers",
        "mark\s*n\s*spencer",
        "marks\s*n\s*spencers",
        "marks\s*spencer",
        "mark\s*spencers",
        "mark\s*spencer",
        "marks\s*spencers",
        "marks\s*\&\s*spencer",
        "mark\s*\&\s*spencers",
        "mark\s*\&\s*spencer",
        "marks\s*\&\s*spencers",
    ]

    line = re.sub('|'.join(ms_versions[::-1]), repl, line.lower())

    ms_versions2 = [
        "\s+m\s?s\s+",
        "\s+m\s?\&\s?s\s+"
    ]
    return re.sub('|'.join(ms_versions2[::-1]), ' ' + repl + ' ', line.lower())


def replace_dates(line, repl=' param_date '):
    """
    Replaces dd/mm/yyyy formatted dates with string in repl.

    :param line: [string], one line of text
    :param repl: [string], string to replace dates with
    :return: [string], where dates are replaced with repl
    """

    date = re.compile(r"\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}")
    return re.sub(date, repl, line)


def replace_postcode(line, repl=' param_postcode '):
    """
    Filters out British postcodes.

    :param line: [string], one line of text
    :param repl: [string], string to replace postcodes with
    :return: [string], where postcodes are replaced with repl
    """
    postcode = re.compile(r"[A-Za-z]{1,2}[0-9Rr][0-9A-Za-z]? [0-9][A-Za-z]{2}")
    return re.sub(postcode, repl, line)


def replace_numbers(line, repl=' param_number ', min_num=2):
    """
    Filters any numeric string.

    :param line: [string], one line of text
    :param repl: [string], string to replace numbers with
    :param min_num: [int], min number of numbers to replace
    :return: [string], where numeric strings are replaced with repl
    """
    return re.sub(r"[\d-]{%d,}" % min_num, repl, line)


def replace_prices(line, repl=' param_price '):
    """
    Filters simple price strings like $32.12

    :param line: [string], one line of text
    :param repl: [string], string to replace prices with
    :return: [string], where price strings are replaced with repl
    """
    price = re.compile(ur"[\$\u20AC\u00A3]{1}\d+\.?\d{0,2}")
    return re.sub(price,  repl, line)


def replace_emails(line, repl=' param_email '):
    """
    Filters out email addresses.

    :param line: [string], one line of text
    :param repl: [string], string to replace email addresses with
    :return: [string], where email addresses are replaced with repl
    """
    return re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", repl, line)


def replace_emoticon(line):
    """
    Filters smileys and sad faces, lols and hahas.

    :param line: [string], one line of text
    :return: [string], where smileys are replaced with param_smileyface.
    """
    line = re.sub(r"lol|hahahaha|hahaha|haha", ' param_funny ', line)
    line = re.sub(r":-?\)+", ' param_smileyface ', line)
    return re.sub(r":-?\(+", ' param_sadface ', line)


def replace_names(line, names, repl=' param_name '):
    """
    Replaces names in a line of text with repl.

    :param line: [string], one line of text
    :param names: [dictionary], with keys being lower case names
    :param repl: [string], string to replace names with
    :return: [string], where English names are replaced with repl string
    """
    line = [repl if w.lower() in names else w for w in line.split(' ')]
    return join_line(line)


def build_names(data_folder, names_file='names.txt'):
    """
    Builds and returns a dictionary from the names file.

    :param data_folder: [string], absolute path to data folder
    :param names_file:  [string], name of the name file
    :return: a dictionary with the names as keys
    """
    names_file = codecs.open(os.path.join(data_folder, names_file), 'rU', 'utf-8')
    names = {}
    for line in names_file:
        names[line.strip()] = 1
    names_file.close()
    return names


def simple_spellcheck(line, word_dict):
    """
    Uses the NTLK words list to remove typos.
    :param line: [string], one line of text
    :param word_dict: [dict], dictionary built from NLTK's words list.
    :return: [string], line with discarded spellchecks
    """
    line_checked = []
    for w in wordpunct_tokenize(line):
        # make sure we don't get rid of param_ms, param_number etc.
        if re.search(r"param_", w):
            line_checked.append(w)
        if w == '||':
            line_checked.append(w)
        # check capitalized version of words as well
        elif w in word_dict or w.capitalize() in word_dict:
            line_checked.append(w)
    return line_checked


def get_contractions():
    contractions = {
        "aren't":  "are not",
        "can't": "cannot",
        "couldn't": "could not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'll": "he will",
        "he's": "he is",
        "i'd": "i would",
        "i'll": "i will",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it's": "it is",
        "let's": "let us",
        "mightn't": "might not",
        "mustn't": "must not",
        "shan't": "shall not",
        "she'd": "she would",
        "she'll": "she will",
        "she's": "she is",
        "shouldn't": "should not",
        "that's": "that is",
        "there's": "there is",
        "they'd": "they would",
        "they'll": "they will",
        "they're": "they are",
        "they've": "they have",
        "we'd": "we would",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "where's": "where is",
        "who'd": "who would",
        "who'll": "who will",
        "who's": "who is",
        "who've": "who have",
        "won't": "will not",
        "wouldn't": "would not",
        "you'd": "you would",
        "you'll": "you will",
        "you're": "you are",
        "you've": "you have"
    }
    return contractions


def replace_contractions(line):
    """
    Replaces the most common contractions in English.
    :param line: [string], one line of text
    :return: [string], line with replaced contactions
    """
    contractions = get_contractions()
    pattern = re.compile(r'\b(' + '|'.join(contractions.keys()) + r')\b')
    line = pattern.sub(lambda x: contractions[x.group()], line)
    return line


def get_ngram_model(lines, i, n):
    """
    Finds the model of the n-th gram (connects phrases into single tokens).
    :param lines: [lList of lists] representation of corpus
    :param i: [int], iterartion
    :param n: [int], number of times the algorithm should run recursively
    :return: [lList of lists], transformed corpus where bigrams are joined by _
    """

    ngram_model = gs.models.Phrases(lines)
    ngram_pred = list(ngram_model[lines])
    if i < n:
        get_ngram_model(ngram_pred, i+1, n)
    return ngram_model


def get_list_of_sentenctes(input_file):
    """
    Returns a list of list where every element is a sentence and on the 2nd
    level every item is a word.
    :param input_file: [file_handler], opened file object
    :return: [list of lists]
    """

    lines = []
    for line in input_file:
        lines.append(line.strip().split())
    return lines


def write_ngrams(data_folder, input_file, n=3):
    """
    Finds bigrams, trigrams, etc from an input file and replaces them with
    phrases that are connected with _. This is saved it to <input_file>Ngram.txt

    :param data_folder: [string], absolute path to data folder
    :param input_file: [string], name of input file
    :param n: [int], number of times the gensim phrases should be applied
    :return: [object], the trained ngram model by gensim
    """

    # open files
    filename, extension = os.path.splitext(input_file)
    input_file = codecs.open(os.path.join(data_folder, input_file), 'rU', 'utf-8')
    if n == 2:
        ngram = 'Bigram'
    elif n == 3:
        ngram = 'Trigram'
    else:
        ngram = 'str(n)gram'
    ngram_file = filename + ngram + extension
    ngram_file = codecs.open(os.path.join(data_folder, ngram_file), 'w', 'utf-8')

    # create list of lists from the lines
    lines = get_list_of_sentenctes(input_file)

    # run gensim's phrase method recursively
    ngram_model = get_ngram_model(lines, 1, n)
    lines = ngram_model[lines]

    # write results
    for line in lines:
        try:
            ngram_file.write(" ".join(line) + "\n")
        except:
            pass

    input_file.close()
    ngram_file.close()