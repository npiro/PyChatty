# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 16:15:27 2016

@author: piromast
"""
# In[1]:
# Import stuff
import os.path
import gensim
import codecs
import numpy as np
import pandas as pd
import warnings
import pickle
# In[2]:
# Load model
folder = '/Users/piromast/Dropbox/S2DS - M&S/Data/'
ModelFileName ='clientW2V_BothSeperate'
model = gensim.models.Word2Vec.load(os.path.join(folder,ModelFileName))

# In[3]:
# Load client text and calculate average word vector
clientFileName = os.path.join(folder, '04_clientMessagesFiltered.txt')

file = codecs.open(clientFileName, 'r', 'utf-8')

#sentence_vects = np.empty((0, 100))
sentence_vects = []

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    for sentence in file:

        vect = np.mean([model[w].T for w in sentence.strip().split() if w in model], axis = 0) # Calculate mean word vector for each sentence
    
        #sentence_vects = np.vstack((sentence_vects,vect))
        #sentence_vects = np.append(sentence_vects, [vect.T], axis = 0) # Calculate mean word vector for each sentence
        sentence_vects.append([vect]) 
         

client_sentences = np.zeros((len(sentence_vects), 100))
for i, sentence in enumerate(sentence_vects):
    client_sentences[i, :] = np.array(sentence)

# In[4]:
# Generate data frame for agent messages and store with pickle
fullAgentMessagesFile = folder + '03_agentMessagesMatch04.txt'
fullClientMessagesFile = folder + '03_clientMessagesMatch04.txt'
with codecs.open(fullAgentMessagesFile, 'r', 'utf-8') as fA, codecs.open(fullClientMessagesFile, 'r', 'utf-8') as fC:
    df = pd.DataFrame({'Agent':[lines for lines in fA], 'Client':[lines for lines in fC]})
with open(folder + 'MessagesDF.pkl', 'wb') as fid:
    pickle.dump(df, fid)

# In[4]:
# Retrieve as follows:
with open(folder + 'MessagesDF.pkl', 'rb') as fid:
    df = pickle.load(fid)
# In [4]:
# Query as follows
df.Client.iloc[0].strip()

# In[5]:
# Define a function to get the vector for a user sentence 
def get_sentence_vector(sentence):
    sentence_matrix = np.array([model[w].T for w in sentence.strip().split() if w in model])
    sentence_vect = np.mean(sentence_matrix, axis=0)
    return sentence_vect

# In[6]:
# Train KD tree

from sklearn.neighbors import KDTree
tree = KDTree(client_sentences)

# In[7}:
# Query tree
dist, ind = tree.query(sentence_vects[0], k=3) 

# In[8}:
# Save tree

# save the classifier
with open(folder + 'kdtree.pkl', 'wb') as fid:
    pickle.dump(tree, fid)

# In[9}:
# load it again
with open(folder + 'kdtree.pkl', 'rb') as fid:
    tree = pickle.load(fid)
    
# In[9]:
# Test
query = 'i want to reschedule my delivery date'
dist, ind = tree.query(get_sentence_vector(query).reshape(1,-1), k=1)
print(dist)
