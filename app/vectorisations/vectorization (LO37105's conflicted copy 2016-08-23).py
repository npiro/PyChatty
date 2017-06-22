"""
Tools to vectorize words, sentences, phrases and documents.

class SentencesGenerator(): iterator class to run through text line by line
class Word2Vec(): Wrapper to gensim word2vec model
class SentenceVectorizer(): sentence vectorizer object

Example code:
    (add app folder to PYTHONPATH)
    from vectorisations.vectorization import Word2Vec, SentenceVectorizer, SearchTree
    
    filename = "../../Data/04_clientMessagesFiltered.txt"
    W2V = Word2Vec(filename)                            # Train word2vec with file in filename
    vectorizer = SentenceVectorizer(W2V)                # make sentence vectorizer object
    matrix = vectorizer.get_sentences_matrix(filename)  # calculate matrix of sentence vectors of whole text file
    tree = SearchTree(matrix)                           # Generate search tree (to find nearest vector)
    
    # Lookup example    
    v = vectorizer.get_sentence_vector("I lost my order")   # Convert sentence to vector
    index, distance = tree.findNearestVector(v)             # find nearest vector
"""

import os.path
import operator
import gensim
import codecs
import numpy as np
import warnings
import pickle
from random import shuffle
import pdb

def remove_NaNs_from(matrix):
    """
    This functions takes the matrix of vectorised words created by Word2Vec.
    It returns as output:
    - matrixFiltered: the matrix of vectorised words created by Word2Vec with the NaN lines removed.
    - correspondanceMap: an array which gives the correspondance map between corresponding lines
    between the matrices "matrix" and "matrixFiltered". correspondanceMatrix[i] returns j where maps
    line i of matrixFiltered corresponds to line j of matrix 
    
    """
    matrixNaN = np.isnan( matrix[:,0] )
    matrixFiltered = matrix[ ~ matrixNaN]    
    
    vectorRange=np.empty( len( matrix[:,1] ) )
    for n in range( len(matrix[:,1]) ):
        vectorRange[n] = n
    correspondanceMap = vectorRange [ ~ matrixNaN ] 
    return matrixFiltered, correspondanceMap.astype(int)



class SentencesGenerator(object):
    """
    class SentencesGenerator(filename):
    
    Iterator class to run through text file line by line.
    The iterator returns a split list of words in the next line in the text file
    Arguments:
        filename: text file name (entire path is necessary)
    """
    def __init__(self, filename):
        self.filename = filename
 
    def __iter__(self):
        for line in codecs.open(self.filename,'r','utf-8'):
            yield line.split()
            
    def to_array(self):
        self.sentences = []
        for i, line in enumerate(codecs.open(self.filename,'r','utf-8')):
            self.sentences.append(line.strip())        
        return self.sentences 
            
class SentencesGeneratorInList(object):
    """
    class SentencesGenerator(filename):
    
    Iterator class to run through text file line by line.
    The iterator returns a split list of words in the next line in the text file
    Arguments:
        filename: text file name (entire path is necessary)
    """
    def __init__(self, filename, sentence_index_list):
        self.filename = filename
        self.sentence_index_list = sentence_index_list
        with codecs.open(filename, 'r', 'utf-8') as f:
            n = 0
            for _ in f:
                n += 1    
            self.num_lines = n
        
        self.is_in_index_list = np.array([False] * self.num_lines)
        self.is_in_index_list[sentence_index_list] = True
        
 
    def __iter__(self):
        for line_num, line in enumerate(codecs.open(self.filename,'r','utf-8')):
            if self.is_in_index_list[line_num]:
                yield line.split()
                
class ParagraphGenerator(object):
    """
    class SentencesGenerator(filename):
    
    Iterator class to run through text file line by line.
    The iterator returns a split list of words in the next line in the text file
    Arguments:
        filename: text file name (entire path is necessary)
    """
    def __init__(self, filename):
        self.filename = filename
 
    def __iter__(self):        
        for i, line in enumerate(codecs.open(self.filename,'r','utf-8')):
            yield gensim.models.doc2vec.TaggedDocument(line.strip(), 'tag'+str(i))
    
    def to_array(self):
        self.sentences = []
        for i, line in enumerate(codecs.open(self.filename,'r','utf-8')):
            self.sentences.append(gensim.models.doc2vec.LabeledSentence(line.strip(), 'tag'+str(i)))        
        return self.sentences 
        
    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences   
        
class Word2Vec(object):
    """
    Wrapped for gensim Word2Vec model
    Arguments:
        filenames: list of text files to use for training
        folder: folder containing text files (default: current folder) 
        load_from_file: set to True to load trained model from <filename> (default = False)
        size: size of word vectors (defaul = 100)
        window: size of word vector training window (default = 5)
        min_count: minimum number of tines a word should appear 
                   to be considered for training (default = 3)
        workers: number of cores to use for training (default = 4)
        
    Usage:
        1) create object specifying text filename as argument.
        2) run train method to train model
        
        or
        1) create object with filename of trained model and load_from_file = True  
    """
    GensimModel = gensim.models.Word2Vec
    Generator = SentencesGenerator
    generator = None
    
    def __init__(self, filenames, folder = '.', load_from_file = False, size = 100, window = 5, min_count = 3, workers = 4):

        self.dictionary = []
        if not load_from_file:
            # Convert single filename string to list with single element
            if isinstance(filenames, str):
                filenames = [filenames]
            # If filenames is actually already a gensim Word2Vec object
            # just save it and return
            elif isinstance(filenames, gensim.models.Word2Vec):
                self.model = filenames
                return
            # Loop through files, create iterator, dictonary and train model.
            # Retrain if more than one file is provided
            for i, fn in enumerate(filenames):
                fn = os.path.join(folder,fn) # combine folder into filename
                generator = self.Generator(fn) # Generate sentence generator object
                self.generator = generator
                self.dictionary.append(gensim.corpora.Dictionary(generator))
                if i == 0:
                    self.model = self.GensimModel(generator, size, window, min_count, workers)
                else:
                    self.model.train(generator, size, window, min_count, workers)
            
        else:
            # If reloading from file, filenames should be a single string
            assert isinstance(filenames, str)   
            self.model = self.load(filenames, folder=folder)
            
            
            
    def save_dictionary(self, filename, folder = '.'):
        """
        Save the word dictonary to filename in folder
        """
        self.dictionary.save(os.path.join(folder,filename))
        
    def retrain(self, filename, folder = '.', size=100, window=5, min_count=3, workers=4):
        """ Retrain model with additional text file"""
        sentencesGenerator = SentencesGenerator(os.path.join(folder,filename))
        self.model.train(sentencesGenerator, size, window, min_count, workers)
        
    def load(self, filename, folder = '.'):
        """ load model from file """
        fn = os.path.join(folder,filename)
        print(fn)
        self.model = self.GensimModel.load(fn)
        
    def save(self, filename, folder = '.'):
        """ save model from file """
        fn = os.path.join(folder,filename)
        self.model.save(fn)
        


class Doc2Vec(Word2Vec):
    """ 
    Wrapper of gensim Doc2Vec class:
    Inherits all methods from Word2Vec
    
    """
    
    def __init__(self, filenames):
        # All that's necessary is to change the Gensim model that's used
        self.GensimModel = gensim.models.Doc2Vec
        self.Generator = ParagraphGenerator
        super(Doc2Vec,self).__init__(filenames)
        sentences = self.generator.to_array()
        self.model.build_vocab(sentences)


class TfIdf(object):
    """
    Encapsulates gensim Tf-idf module:
    constructor arguments:
        filename: text file with one line per "document" or chat in our case
        folder: (optional) location of file (defaul = '.')
        
        Usage:
        Instantiate with filename and folder (optional)
        Call GetWordWeight method with a string as argument to get the weight
    """
    def __init__(self, filename, folder = '.'):
        """
        Initializes TfIdf object. 
        Specify traiing text filename and folder (optionally)
        Attributes:
            - model: Tfidf vectorizer object (apply to word to obtain vector)
            - matrix: full Tfidf matrix (rows: documents, columns: words)
            - WordWeightDict: dictonary containing relative weight of each
                    word (the weight is the inverse of the idf = df)
            - MaxIDF: Maximum IDF value (used to normalize idf)
            
        """
        fn = os.path.join(folder,filename) # combine folder into filename
        from sklearn.feature_extraction.text import TfidfVectorizer
        with codecs.open(fn,'r','utf-8') as file:
            self.model = TfidfVectorizer()
            self.matrix = self.model.fit_transform(file)
            idf = self.model.idf_
            self.WordWeightDict = dict(zip(self.model.get_feature_names(), idf))
            self.MaxIDF = max(idf)
            self.WordWeightDict.update((x, y/self.MaxIDF) for x, y in self.WordWeightDict.items())

            
    def GetWordWeight(self, word):
        """
        Get the tf-idf weight of word
        """
        if word in self.WordWeightDict:
            return self.WordWeightDict[word.lower()]
        else:
            return 0.0

    
    
    def GetLargestWeightItem(self):
        """
        Get the word (and its weight) with the largest weight
        """
        return max(self.WordWeightDict.items(), key=operator.itemgetter(1))
        
    def GetSmallestWeightItem(self):
        """
        Get the word (and its weight) with the smallest weight
        """
        
        return min(self.WordWeightDict.items(), key=operator.itemgetter(1))

class SentenceVectorizer(object):
    """
    Class to compute sentence vectors. 
    Public methods:
    get_sentence_vector(sentence): get sentence vector
    
    get_sentences_matrix(filename, folder): get sentences matrix
    
    Usage:
    1) Generate Word2Vec object: W2V = Word2Vec(filename)
    2) Generate SentenceVectorizer object passing W2V: sv = SentenceVectorizer(W2V)
    3.1) Vectorize sentence with sv.get_sentence_vector(sentence)
    3.2) Vectorize sentences with sv.get_sentences_matrix(filename)
    
    """
    def __init__(self, W2V, filename = None, method = 'average'):
        """ Initialize with
            model: Trained word2vec model of class vectorization.Word2Vec
            method (optional): Sentence vectorization method. Default = 'average' (average of word vectors)
        """
        self.W2V = W2V
        self.method = method
    
        if method == 'average':
            self.get_sentence_vector = self.get_average_vector
        elif method == 'idf-weighted-average':
            self.get_sentence_vector = self.get_idf_weighted_average_vector
            assert(filename is not None)
            self.tf = TfIdf(self,filename)
    
    def get_d2v_vector(self,D2V):
        pass
    
    def get_idf_weighted_average_vector(self,sentence):
        """
        Return the average word vector of all words in sentence.
        Arguments:
            sentence: sentence to vectorize
        """
        if isinstance(sentence, str):
            sentence = sentence.strip().split()
        sentence_matrix = np.array([self.W2V.model[w].T for w in sentence if w in self.W2V.model])
        sentence_vect = np.mean(sentence_matrix, axis=0)
        return sentence_vect
        
    def get_average_vector(self,sentence):
        """
        Return the average word vector of all words in sentence.
        Arguments:
            sentence: sentence to vectorize
        """
        if isinstance(sentence, str) or isinstance(sentence, unicode):
            sentence = sentence.strip().split()
        sentence_matrix = np.array([self.W2V.model[w].T for w in sentence if w in self.W2V.model])
        sentence_vect = np.mean(sentence_matrix, axis=0)
        return sentence_vect

    def get_sentences_matrix(self, filename=None, sentenceGenerator=None, folder = '.'):     
        """ 
        Return the matrix of word-vectors representing each sentence in file
        Assumes that there is one sentence per line
        Arguments:
            filename: file to vectorize (omit to use sentenceGenerator)
            sentenceGenerator: SentenceGenerator object to use to extract sentences from file
            folder (optional): folder containing file (default = '.')
        """

        sentence_vects = []
        if filename is not None:
            filename = os.path.join(folder, filename)
            with codecs.open(filename, 'r', 'utf-8') as file:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    
                    for sentence in file:
                        vect = self.get_sentence_vector(sentence)
                        sentence_vects.append([vect]) 
        
        elif sentenceGenerator is not None:
            for sentence in sentenceGenerator:
                vect = self.get_sentence_vector(sentence)
                sentence_vects.append([vect]) 
        
        else:
            ValueError('Error in get_sentences_matrix: need either a filename or a sentenceGenerator object')
   
 
        sentences_matrix = np.zeros((len(sentence_vects), len(sentence_vects[0][0])))
        for i, sentence in enumerate(sentence_vects):
            sentences_matrix[i, :] = np.array(sentence)
                
        return sentences_matrix

from sklearn.neighbors import KDTree

class SearchTree(object):
    """
    KD tree data structure to search for nearest vector
    Usage:
        SearchTree(sentences): to create tree from sentence vectors
        SearchTree(filename=filename, folder=folder): to load from a file in folder
    """
    def __init__(self, sentences_matrix=None, filename=None, folder='.'):
        if sentences_matrix is not None:
            self.tree = KDTree(sentences_matrix)
        else:    
            self.load_tree(filename, folder=folder)
    
    def save_tree(self, filename, folder='.'):
        """
        Saves tree to filename at folder
        """
        
        with open(os.path.join(folder,filename), 'wb') as fid:
            pickle.dump(self.tree, fid)
            
    def load_tree(self, filename, folder='.'):
        """
        Loads tree from filename in folder
        """
        with open(os.path.join(folder,filename), 'rb') as fid:
            self.tree = pickle.load(fid)
    
    def findNearestVector(self,vector):
        """
        Find vector in tree nearest to passed vector.
        Returns: 
            ind: index of nearest vector
            dist: distance of nearest vector
        """
        dist, ind = self.tree.query(vector.reshape(1,-1), k=1)
    
        return ind[0][0], dist[0][0]

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 17:36:05 2016
This function returns the matrix of vectorised words created by Word2Vec, but without the lines with a NaN. Additionally, it 
returns an array with gives the correspondance between the line indexes of this new matrix without NaNs and the new
@author: Jean-Christophe
"""
import numpy as np
    


if __name__ == "__main__":
    # Usage example
    # don't forget to import vectorization
    filename = os.path.join("..","..","Data","04_clientMessagesFiltered.txt")
    W2V = Word2Vec(filename)    # Train word2vec with file in filename
    vectorizer = SentenceVectorizer(W2V)    # make sentence vectorizer object
    matrix = vectorizer.get_sentences_matrix(filename)  # calculate matrix of sentence vectors of whole text file
    tree = SearchTree(matrix)   # Generate search tree (to find nearest vector)
    
    # Lookup example    
    v = vectorizer.get_sentence_vector("I lost my order")   # Convert sentence to vector
    index, distance = tree.findNearestVector(v)             # find nearest vector
    