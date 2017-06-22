
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 23:29:55 2016

@author: N.Piro
"""

from vectorisations.vectorization import Word2Vec, SentenceVectorizer, SearchTree
import message_db.message_db as mdb

import pandas as pd
import numpy as np
import os
import pickle

class ChattyV1(object):
    def __init__(self, folder = 'C:\\Users\\Client\\Dropbox\\S2DS - M&S\\Data', 
                 filenameClient = '04_clientMessagesFiltered.txt', 
                 filenameAgent = '04_agentMessagesFiltered.txt',
                 filenameMessageDF = 'MessagesDF.pkl',
                 filenameW2V = 'W2V_Chatty1_0.mdl', 
                 filenameSearchTree = 'SearchTree_Chatty1_0.pkl',
                 filenameMessageMap = 'Filtered2UnfilteredMessageMap_Chatty1_0.pkl',
                 load_from_file = True):
        
        # Init bot        
        self._initChatBot(folder, filenameClient, filenameAgent,
                          filenameW2V, filenameSearchTree, filenameMessageMap,
                          load_from_file)
            
    def _initChatBot(self, folder, filenameClient, filenameAgent, 
                     filenameW2V, filenameSearchTree, filenameMessageMap,
                     load_from_file, retrain_on_agent_file = False):
        
        # Load messages dataframe
        self.MessagesDataFrame = mdb.load_data_frame('MessagesDF.pkl',folder)   
        fn = os.path.join(folder, filenameMessageMap)
        with open(fn, 'rb') as fid:                       
            self.FilteredToUnfilteredMessageMap = pickle.load(fid)

        # Either load everything from files:
        if load_from_file:
            self.W2V = Word2Vec(filenameW2V, folder = folder, 
                                load_from_file = True)
                                
            self.tree = SearchTree(sentences_matrix=None, 
                                   filename=filenameSearchTree, 
                                   folder=folder)
            

        
        # or generate it from scratch:
        else:
            # Train W2V model on client file
            self.W2V = Word2Vec(filenameClient, folder = folder, workers = 7)
            
            # Retrain (optionally) on agent file
            if retrain_on_agent_file:
                self.W2V.retrain(filenameAgent, folder = folder)
            
            # Make the sentence vectorizer object, get sentence matrix
            # and generate search tree
            
            self.tree = SearchTree(self.matrix)
            
        self.vectorizer = SentenceVectorizer(self.W2V)    
        self.matrix = self.vectorizer.get_sentences_matrix(filenameClient, 
                                                               folder = folder)
    
    def getReplyFromChatBot(self, input_message):
        try:
            # Vectorize message
            vect = self.vectorizer.get_sentence_vector(input_message)
            print(vect)
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
            print type(inst)     # the exception instance
            print inst.args      # arguments stored in .args
            print inst           # __str__ allows args to be printed directly
