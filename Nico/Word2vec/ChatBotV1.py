# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 19:27:30 2016

@author: piromast
"""

import os.path
import gensim
import codecs
import numpy as np
import pickle


folder = 'C:\\Users\\Client\\Dropbox\\S2DS - M&S\\Data'
ModelFileName ='clientW2V_BothSeperate'

# Retrieve client tree
with open(os.path.join(folder, 'kdtree.pkl'), 'rb') as fid:
    tree = pickle.load(fid)
    
# Retrieve messages data frame:
with open(os.path.join(folder, 'MessagesDF.pkl'), 'rb') as fid:
    df = pickle.load(fid)

# Retrieve word2vec model
model = gensim.models.Word2Vec.load(os.path.join(folder,ModelFileName))
    

# Define a function to get the vector for a user sentence 
def get_sentence_vector(sentence):
    sentence_matrix = np.array([model[w].T for w in sentence.lower().strip().split() if w in model])
    sentence_vect = np.mean(sentence_matrix, axis=0)
    return sentence_vect


def get_message(line_num):

    ClientText = df.Client.iloc[line_num].strip()
    AgentText = df.Agent.iloc[line_num].strip()
        
    return (ClientText, AgentText)
            

# In[9}:
# Test


while True:
    query = input('<Chatty 1.0> ')
    if query == 'quit':
        break
    try:
        vect = get_sentence_vector(query)
    
        dist, ind = tree.query(vect.reshape(1,-1), k=1)
    
        Input, Output = get_message(ind[0][0])
        print('Nearest user sentence:'+Input)
        print(Output)
    except:
        print("I don't understand. Please be more clear! ")
