# -*- coding: utf-8 -*-
"""
This function returns a matrix representing the coordinates of the most frequent words in the vectors space after a Word2Vec analysis.
=======
Created on Wed Aug 10 16:36:54 2016


@author: Jean-Christophe
"""

import gensim
import numpy as np
import codecs

def importMatrixWord2Vec(ModelFileName, UrFileName, words_number, numDim):
    """
    Created on Wed Aug 10 16:36:54 2016
    This function takes into input:
    - ModelFileName: path to the file storing the corresponding Word2Vec table
    - UrFileName: path to the "Ur file" (from the CCA analysis) corresponding to the same corpus of text that Word2Vec was applied to
    - words_number: the number of words required (the words are ranked by frequency, so if words_number = 50 it will the return the 50th most used words of the corpus)
    - numDim: the dimension of the vectors space representing the words, in Word2Vec
    
    The function returns an array of dimension words_number * numDim, representing the coordinates in vector space of the words_number^th most frequent words of the corpus
    
    """
    
    model = gensim.models.Word2Vec.load(ModelFileName)
    urFile=codecs.open(UrFileName,'r','utf-8')


    vectors = np.empty([words_number, numDim])
    words = list()
    urFile.readline()

    for i in range(words_number):
        L1list = urFile.readline()
        [frequency, word, dump] = L1list.split(' ',2)
        words.append(word)
        for j in range(numDim):
            vectors[i,j] = model[word][j]
    
    return vectors
            
#importMatrixWord2Vec('clientW2V_BothSeperate', 'Ur', 1000, 100)