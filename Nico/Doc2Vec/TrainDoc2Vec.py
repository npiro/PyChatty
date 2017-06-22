# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 16:42:36 2016

@author: piromast
"""
# In[1]:
import os.path
from vectorisations.vectorization import Doc2Vec, SentenceVectorizer, SearchTree, remove_NaNs_from

# In[2]:
filename = os.path.join("/Users","piromast","Dropbox","S2DS - M&S","Data","04_doc2vecTrainingDataFilteredTrigramLemmatized.txt")
D2V = Doc2Vec(filename)    # Train word2vec with file in filename
vectorizer = SentenceVectorizer(D2V)    # make sentence vectorizer object
matrix = vectorizer.get_sentences_matrix(filename)  # calculate matrix of sentence vectors of whole text file

# In[3]:
# Remove nans
matrix,_ = remove_NaNs_from(matrix)

# In[4]:
# K means clustering
from sklearn.cluster import KMeans
km=KMeans(n_clusters = 15)
km.fit(matrix)

# In[4]:
# Build search tree
tree = SearchTree(matrix)   # Generate search tree (to find nearest vector)

# In[5]:
# Lookup example    
v = vectorizer.get_sentence_vector("I lost my order")   # Convert sentence to vector
index, distance = tree.findNearestVector(v)             # find nearest vector
