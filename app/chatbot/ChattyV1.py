# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 23:29:55 2016

@author: N.Piro
"""

from vectorisations.vectorization import Word2Vec, SentenceVectorizer, SearchTree, remove_NaNs_from
import message_db.message_db as mdb

import pandas as pd
import numpy as np
import os
import pickle
import scipy as sp
from preprocessing.clean_input import InputCleaner

class ChattyV1(object):
                          
    def __init__(self, folder = 'C:\\Users\\Client\\Dropbox\\S2DS - M&S\\Data', 
                 filenameClient = '05_clientMessagesFilteredfastText.txt', 
                 filenameAgent = '05_agentMessagesFilteredfastText.txt',
                 filenameMessageDF = 'MessagesDF.pkl',
                 filenameW2V = 'W2V_Chatty1_0.mdl', 
                 filenameSearchTree = 'SearchTree_Chatty1_0.pkl',
                 filenameMessageMap = 'Filtered2UnfilteredMessageMap_Chatty1_0.pkl',
                 filenameMessageInfoTable = 'client_agent_summary3.csv',
                 filenameMatrix = 'SentenceMatrix_Chatty1_0.pkl',
                 load_from_file = True):
        
        # Init properties
        self.name = 'Chatty 1.0'
        self.folder = folder
        self.filenameClient = filenameClient
        self.filenameAgent = filenameAgent
        self.filenameMessageDF = filenameMessageDF
        self.filenameW2V = filenameW2V
        self.filenameSearchTree = filenameSearchTree
        self.filenameMessageMap = filenameMessageMap
        self.filenameMessageInfoTable = filenameMessageInfoTable
        self.filenameMatrix = filenameMatrix
        
        # Init bot        
        self._initChatBot(folder, filenameClient, filenameAgent, 
                          filenameMessageDF, filenameMessageInfoTable, load_from_file, retrain_on_agent_file = True)
        
            
    def _initChatBot(self, folder, filenameClient, filenameAgent, filenameMessageDF,
                     filenameMessageInfoTable, load_from_file, retrain_on_agent_file = False):

        # Load messages dataframe
        self.MessagesDataFrame = mdb.load_data_frame(filenameMessageDF, folder) 
            
        # Import chat info table
        filenameIndexTable = os.path.join(folder,filenameMessageInfoTable)
        indexTable = pd.read_csv(filenameIndexTable)
        # Extract sentence level information and conversation number
        # (the order of each sentence in a conversation)
        self.sentenceLevel = indexTable['convPos'].values.reshape((-1,1))
        self.conversationNumber = indexTable['convID'].values.reshape((-1,1))
        linePositionInFilteredFile = indexTable['linePosFiltered'].values
        linePosition = indexTable['linePos'].values
        self.sentenceLevelNoNans = self.sentenceLevel[~np.isnan(linePositionInFilteredFile)]
        
        # Either load everything from files:
        if load_from_file:
            self.load()
        
        # or generate it from scratch:
        else:
            # Train W2V model on client file
            self.W2V = Word2Vec(filenameClient, folder = folder, workers = 7)
            
            # Retrain (optionally) on agent file
            if retrain_on_agent_file:
                self.W2V.retrain(filenameAgent, folder = folder, workers = 7)
            
            # Make the sentence vectorizer object, get sentence matrix
            # and generate search tree
            self.vectorizer = SentenceVectorizer(self.W2V)
            self.sentenceMatrix = self.vectorizer.get_sentences_matrix(filenameClient, 
                                                               folder = folder)
            
            # Construct map from filtered to unfiltered file line number 
            self.FilteredToUnfilteredMessageMap = [np.int(linePosition[i]) 
            for i in linePosition.tolist() if not np.isnan(linePositionInFilteredFile[i])]
                    
            # Normalize sentenceMatrix
            self.sentenceMatrixNoNans, self.correspondenceVector = remove_NaNs_from(self.sentenceMatrix)
            #self.MaximumVectorLength = np.max(sp.linalg.norm(self.sentenceMatrixNoNans,ord=2,axis=1))
            #self.sentenceMatrixNormalized = self.sentenceMatrixNoNans/self.MaximumVectorLength

            # Extend sentence matrix with one dimention equal to level
            #self.sentenceMatrixExtended=np.hstack((self.sentenceMatrixNormalized,self.sentenceLevelNoNans[self.correspondenceVector]*self.levelFactor))
            
            # Calculate compound filtered to unfiltered message map
            self.FilteredToUnfilteredMessageMap = [self.FilteredToUnfilteredMessageMap[i] for i in self.correspondenceVector]
            

            self.tree = SearchTree(self.sentenceMatrixNoNans)

        # Instantiate input cleaner object 
        self.cleaner = InputCleaner(folder)
        
        
    def getReplyFromChatBot(self, input_message):
        try:
            # Clean message
            input_message = self.cleaner.clean_input(input_message)
            
            # Vectorize message
            vect = self.vectorizer.get_sentence_vector(input_message)
            
            # Find nearest message in client corpus
            ind, _ = self.tree.findNearestVector(vect)
            print(ind)
            
            # Unmap to account for filter-removed messages
            ind = self.FilteredToUnfilteredMessageMap[ind]
    
            # Get closest client and agent message
            df = self.MessagesDataFrame
            ClientMessage = mdb.get_client_message_from_dataframe(df, ind)
            AgentMessage = mdb.get_agent_message_from_dataframe(df, ind)
            return (AgentMessage, ClientMessage)
            
        except Exception as inst:
            print(type(inst))    # the exception instance
            print(inst.args)     # arguments stored in .args
            print(inst)           # __str__ allows args to be printed directly

    def save(self, folder = None, filenameW2V = None, filenameMatrix = None, 
             filenameSearchTree = None, filenameMessageMap = None):
        if folder is None:
            folder = self.folder
        if filenameW2V is None:
            filenameW2V = self.filenameW2V
        if filenameMatrix is None:
             filenameMatrix = self.filenameMatrix
        if filenameSearchTree is None:
             filenameSearchTree = self.filenameSearchTree
        if filenameMessageMap is None:
            filenameMessageMap = self.filenameMessageMap

        self.W2V.save(filenameW2V, folder)
        print('saved W2V')
        self.tree.save_tree(filenameSearchTree, folder)
        print('saved tree')

        with open(os.path.join(folder,filenameMessageMap),'wb') as f:
            pickle.dump(self.FilteredToUnfilteredMessageMap, f, protocol=2)
        print('saved message map')

        with open(os.path.join(folder,filenameMatrix),'wb') as f:
            pickle.dump(self.sentenceMatrix, f, protocol=2)   
        print('saved matrix')

    def load(self, folder = None, filenameW2V = None, filenameMatrix = None, 
             filenameSearchTree = None, filenameMessageMap = None):
        if folder is None:
            folder = self.folder
        if filenameW2V is None:
            filenameW2V = self.filenameW2V
        if filenameMatrix is None:
             filenameMatrix = self.filenameMatrix
        if filenameSearchTree is None:
             filenameSearchTree = self.filenameSearchTree
        if filenameMessageMap is None:
            filenameMessageMap = self.filenameMessageMap
        # Load w2v object


        self.W2V = Word2Vec(filenameW2V, folder = folder, 
                                load_from_file = True)
        print('loaded W2V')
        
        # Load vectorizer         

        self.vectorizer = SentenceVectorizer(self.W2V)
        print('loaded vectorizer')
        
        # Load search tree
        self.tree = SearchTree(sentences_matrix=None, 
                                   filename=filenameSearchTree, 
                                   folder=folder)
        print('loaded tree')
        
        # Load sentence matrix and prepare submatrices                           
        fn = os.path.join(folder, filenameMatrix)
        with open(fn, 'rb') as fid:                       
            self.sentenceMatrix = pickle.load(fid)
            self.sentenceMatrixNoNans, self.correspondenceVector = remove_NaNs_from(self.sentenceMatrix)
            # Extend sentence matrix with one dimention equal to level
            
            print('loaded matrix')
        # Load message map
        fn = os.path.join(folder, filenameMessageMap)
        with open(fn, 'rb') as fid:                       
            self.FilteredToUnfilteredMessageMap = pickle.load(fid)
            print('loaded message map')
        
    def resetChat(self):
        return
    
    def changeLevelFactor(self, levelFactor):
        return