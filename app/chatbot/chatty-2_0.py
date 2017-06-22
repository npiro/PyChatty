# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 16:52:19 2016

@author: piromast
"""
# In[1]:
from vectorisations.vectorization import Word2Vec, SentenceVectorizer, SearchTree, remove_NaNs_from
import message_db.message_db as mdb
import os
from preprocessing.filtering import filter_line, build_names, NLTKPreprocessor

folder = 'C:\\Users\\Client\\Dropbox\\S2DS - M&S\\Data'
#filenameAgent = '04_agentMessagesFilteredTrigramLemmatized.txt'
filenameClient = '04_clientMessagesFilteredOld.txt'
filenameAgent = '04_agentMessagesFilteredOld.txt'

W2V = Word2Vec(filenameClient, folder = folder)

vectorizer = SentenceVectorizer(W2V)
sentenceMatrix = vectorizer.get_sentences_matrix(filenameClient, folder = folder)

tree = SearchTree(sentenceMatrix)

# In[2]:
import pandas as pd
import numpy as np
import scipy as sp

filenameIndexTable = os.path.join(folder,'client_agent_summaryOld.csv')

indexTable = pd.read_csv(filenameIndexTable)
sentenceLevel = indexTable['convPos'].values.reshape((-1,1))
conversationNumber = indexTable['convID'].values.reshape((-1,1))
linePositionInFilteredFile = indexTable['linePosFiltered'].values
linePosition = indexTable['linePos'].values
# Construct map from filtered to unfiltered file line number 
filteredToUnfiltedPositionMap = [np.int(linePosition[i]) for i in linePosition.tolist() if not np.isnan(linePositionInFilteredFile[i])]
sentenceLevelNoNans = sentenceLevel[~np.isnan(linePositionInFilteredFile)]

# In[3]:
from sklearn.preprocessing import normalize
import scipy as sp

LevelFactor = 0.5
# Normalize sentenceMatrix
sentenceMatrixNoNans, correspondenceVector = remove_NaNs_from(sentenceMatrix)
MaximumVectorLength = np.max(sp.linalg.norm(sentenceMatrixNoNans,ord=2,axis=1))
sentenceMatrixNormalized = sentenceMatrixNoNans/MaximumVectorLength
filteredToUnfiltedPositionMap = [filteredToUnfiltedPositionMap[i] for i in correspondenceVector]
            
# Extend sentence matrix with one dimention equal to level
sentenceMatrixExtended=np.hstack((sentenceMatrixNormalized,sentenceLevelNoNans[correspondenceVector]*LevelFactor))

treeExtended = SearchTree(sentenceMatrixExtended)
# 

# In[2]:
df = mdb.load_data_frame('MessagesDF.pkl',folder)

from preprocessing.clean_input import InputCleaner



level = 0
while True:
    query = raw_input('<Chatty 2.0> ').decode("utf-8")

    cleaner = InputCleaner(folder)
    # Apply preprocessing to input query
    query = cleaner.clean_input(query)
    
    print('Filtered query:'+query)
    if query == 'quit':
        break
    try:
        vect = vectorizer.get_sentence_vector(query)/MaximumVectorLength
        vect = np.hstack((vect,level*LevelFactor))
        ind, _ = treeExtended.findNearestVector(vect)
        # account for filter-out lines
        #print(ind)
        ind = correspondenceVector[ind]
        ind = filteredToUnfiltedPositionMap[ind]  
        #print(ind)
        ClientMessage = mdb.get_client_message_from_dataframe(df, ind)
        AgentMessage = mdb.get_agent_message_from_dataframe(df, ind)
        print('Nearest user sentence: \n'+ClientMessage)
        print('Reply: \n'+AgentMessage)
        level += 1
    except:
        print("I don't understand. Please be more clear! ")

# In[3]:
import pickle

W2V.save('W2V_Chatty2_0.mdl', folder)

tree.save_tree('SearchTree_Chatty2_0.pkl', folder)

with open(os.path.join(folder,'Filtered2UnfilteredMessageMap_Chatty2_0.pkl'),'wb') as f:

    pickle.dump(filteredToUnfiltedPositionMap, f, protocol=2)
