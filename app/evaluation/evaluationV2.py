# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 17:18:10 2016

@author: piromast
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
#from builtins import *
import os, codecs

from sklearn.cross_validation import KFold
import gensim

from vectorisations.vectorization import SentencesGeneratorInList, Word2Vec, SentenceVectorizer, SearchTree, remove_NaNs_from

import numpy as np
import scipy.spatial.distance as spd

import matplotlib.pyplot as plt

import pandas as pd

class ChatEvaluator(object):
    """
    Class to perform evaluation of chat bot conversations
    train(trainingSetFilename): 
        train Word2Vec with specified training set (trainingSetFilename)
    evaluate(crossValidationSetFilename): 
        evaluate by using cross validation set (crossValidationSetFilename)
        calculating mean cosine similarity
    """
    def __init__(self, inputFileNames, outputFileName, filenameMessageInfoTable):
        if type(inputFileNames) is not list:
            inputFileNames = [inputFileNames]
        self.inputFileNames = inputFileNames
        self.outputFileName = outputFileName
        self.filenameMessageInfoTable = filenameMessageInfoTable
        
        self.num_lines = []
        for fn in self.inputFileNames:
            with codecs.open(fn, 'rU', 'utf-8') as f:
                self.num_lines.append(sum(1 for _ in f))
    
    def test(self, num_splits = 3, size = 100, window = 5, min_count = 3, workers = 7, levelFactor = 1.0):
        """
        Perform cross-validation test. 
        1) Split text data in two parts, in num_splits different ways
        
        2) Train using a subset of inputFileNames with:
            - num_lines: number of text lines to use
            - 
            - size: dimensions of word vectors to train
            - window: word2vec windowing size
            - min_count: min number of counts per word to consider for training
            - workers: number of cpu cores to use for training
            
        3) Try predicting testing set input. Compare predicted output to
           expected output. Calculates three different measures:
           input and output eucledian distances and cosine similarity of 
           expected and predicted output vectors. Averages over entire test set
        """
        
        kf = KFold(self.num_lines[0], n_folds=num_splits)
        
        scoreDistInput, scoreDistOutput, scoreCosSimil = [], [], []
        
        for train_index, test_index in kf:
            print("TRAIN:", train_index, "TEST:", test_index,'\n')
            sdin, sdout, sc = self.__get_score(train_index, test_index, size, window, min_count, workers)
            scoreDistInput.append(sdin)
            scoreDistOutput.append(sdout)
            scoreCosSimil.append(sc)
        
        return (scoreDistInput, scoreDistOutput, scoreCosSimil)
        
    # Private methods:
    def __get_score(self, train_index, test_index, size, window , min_count, workers, levelFactor):
        """
        Private method doing the hard work described in test method
        """
        filenameIndexTable = os.path.join(folder,self.filenameMessageInfoTable)
        indexTable = pd.read_csv(filenameIndexTable)
        # Extract sentence level information and conversation number
        # (the order of each sentence in a conversation)
        sentenceLevel = indexTable['convPos'].values.reshape((-1,1))
        
        for i, inputFile in enumerate(self.inputFileNames):
            print('Training iteration:',str(i))
            sentenceGeneratorTrainSetInput = SentencesGeneratorInList(inputFile, train_index)
            if i == 0:  # First train is done creating instance of Word2Vec 
                
                W2VInput = gensim.models.Word2Vec(sentenceGeneratorTrainSetInput, size, window, min_count, workers)
                W2VInput = Word2Vec(W2VInput)
            else:       # Subsequent training is done calling train method
                W2VInput.model.train(sentenceGeneratorTrainSetInput, size, window, min_count, workers)
        
        
        # Generate input and output sentence generators
        sentenceGeneratorTestSetInput = SentencesGeneratorInList(self.inputFileNames[0], test_index)
        sentenceGeneratorTestSetOutput = SentencesGeneratorInList(self.outputFileName, test_index)
        
        # Train input and output word2vec models
        W2VOutput = gensim.models.Word2Vec(sentenceGeneratorTestSetOutput, 
                                           size, window, min_count, workers)
        W2VOutput = Word2Vec(W2VOutput)
        vectorizerInput = SentenceVectorizer(W2VInput)
        vectorizerOutput = SentenceVectorizer(W2VOutput)
        
        # calculate matrix of sentence vectors of whole text file
        matrixInput = vectorizerInput.get_sentences_matrix(sentenceGenerator = 
        sentenceGeneratorTestSetInput)
        matrixInputExtended=np.hstack((matrixInput, sentenceLevel*levelFactor))        
        
        matrixInput,_ = remove_NaNs_from(matrixInputExtended)        
        
        matrixOutput = vectorizerOutput.get_sentences_matrix(sentenceGenerator = 
        sentenceGeneratorTestSetOutput)
        matrixOutputExtended=np.hstack((matrixOutput, sentenceLevel*levelFactor)) 
        matrixOutput,_ = remove_NaNs_from(matrixOutputExtended)         
        
        # Generate search tree (to find nearest vector)
        treeInput = SearchTree(matrixInput)   
        #treeOutput = SearchTree(matrixOutput)   
        
        n = 0 # Sentence number counter     
        
        # Initialize different distance measures 
        # (input and output eucledian distance and cos similarity)
        total_input_distance = 0  
        total_output_distance = 0
        total_cosine_similarity = 0
        
        for vIn, vOut in zip(matrixInput, matrixOutput):
            #print('Print:',str(i))
            n += 1
            # Find nearest vector to input in sentence corpus
            indexInput, distanceInput = treeInput.findNearestVector(vIn) 
            
            # Accumulate input vector distance to nearest neighbout
            total_input_distance += distanceInput
            
            # Calculating output vector corresponding 
            # to nearest input sentence
            vOutPredicted = matrixOutput[indexInput,]
            
            # Caluclating eucledian distance 
            # between observed and predicted output 
            distanceOutput = spd.euclidean(vOut, vOutPredicted)
            total_output_distance += distanceOutput 
            
            # Calculating cosine similarity between observation and prediction
            cosineSimilarity = spd.cosine(vOut, vOutPredicted)
            total_cosine_similarity += cosineSimilarity
            
        return (total_cosine_similarity, 
                total_input_distance, total_output_distance)

                
if __name__ == "__main__":
    folder = 'D:\\Dropbox\\S2DS - M&S\\Data'
    filenameAgent = '04_agentMessagesFiltered.txt'
    filenameAgent = os.path.join(folder,filenameAgent)
    filenameClient = '04_clientMessagesFiltered.txt'
    filenameClient = os.path.join(folder,filenameClient)
    filenameMessageInfoTable = 'client_agent_summary2.csv'
    evaluator = ChatEvaluator(filenameClient, filenameAgent, filenameMessageInfoTable)
    
    evaSize = range(100, 120, 10)
    evaWin = range(2, 4, 1)
    evaMin = range(10, 120, 10)
    Cos_simils = np.zeros((len(evaSize), len(evaWin), len(evaMin)))
    Input_dists = np.zeros((len(evaSize), len(evaWin), len(evaMin)))
    Output_dists = np.zeros((len(evaSize), len(evaWin), len(evaMin)))
    
    
    for i in range(len(evaSize)):
        for j in range(len(evaWin)):
            for k in range(len(evaMin)):
                cosSim, distIn, distOut = evaluator.test(size = evaSize[i], min_count = evaMin[k], window = evaWin[j], levelFactor = 1.0 )
                Cos_simils[i, j, k] = np.average(cosSim)
                Input_dists[i, j, k] = np.average(distIn)
                Output_dists[i, j, k] = np.average(distOut)

    X, Y = np.meshgrid(evaSize, evaWin)
    
    for k in range(len(evaMin)):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(X, Y, Cos_simils[:, :, k] , rstride=10, cstride=10)
        plt.ylabel("Window size")
        plt.xlabel("Vect. space dimension")
        plt.title("Cosine similarity")
        plt.show()
    
    for k in range(len(evaMin)):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(X, Y,  Input_dists[:, :, k] , rstride=10, cstride=10)
        plt.ylabel("Window size")
        plt.xlabel("Vect. space dimension")
        plt.title("Input distance")
        plt.show()

    for k in range(len(evaMin)):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(X, Y, Output_dists[:, :, k] , rstride=10, cstride=10)
        plt.ylabel("Window size")
        plt.xlabel("Vect. space dimension")
        plt.title("Output distance")
        plt.show()

    
    