# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 16:52:19 2016

@author: piromast
"""
# In[1]:
from vectorisations.vectorization import Word2Vec, SentenceVectorizer, SearchTree
import message_db.message_db as mdb

folder = 'C:\\Users\\Client\\Dropbox\\S2DS - M&S\\Data'
filenameClient = '04_clientMessagesFiltered.txt'
filenameAgent = '04_agentMessagesFiltered.txt'
W2V = Word2Vec(filenameClient, folder = folder, workers = 7)

#W2V.retrain(filenameAgent, folder = folder)

vectorizer = SentenceVectorizer(W2V)
matrix = vectorizer.get_sentences_matrix(filenameClient, folder = folder)
tree = SearchTree(matrix)

# In[2]:
import pandas as pd
import numpy as np
import os

filenameIndexTable = os.path.join(folder,'client_agent_summary.csv')

indexTable = pd.read_csv(filenameIndexTable)

linePositionInFilteredFile = indexTable['linePosFiltered'].values
linePosition = indexTable['linePos'].values
# Construct map from filtered to unfiltered file line number 
filteredToUnfiltedPositionMap = [np.int(linePosition[i]) for i in linePosition.tolist() if not np.isnan(linePositionInFilteredFile[i])]


# In[2]:
df = mdb.load_data_frame('MessagesDF.pkl',folder)

while True:
    # Careful with this. Python 3 uses input and directly converts to utf-8. Python 2 doesn't
    query = raw_input('<Chatty 1.0> ').decode("utf-8") 
    print(query)
    if query == 'quit':
        break
    try:
        vect = vectorizer.get_sentence_vector(query)
        
        ind, _ = tree.findNearestVector(vect)
        ind = filteredToUnfiltedPositionMap[ind]
        print(ind)
        ClientMessage = mdb.get_client_message_from_dataframe(df, ind)
        AgentMessage = mdb.get_agent_message_from_dataframe(df, ind)
        print('Nearest user sentence:'+ClientMessage)
        print('Answer:'+AgentMessage)
    except:
        print("I don't understand. Please be more clear! ")

# In[3]:
import pickle

W2V.save('W2V_Chatty1_0.mdl', folder)

tree.save_tree('SearchTree_Chatty1_0.pkl', folder)

with open(os.path.join(folder,'Filtered2UnfilteredMessageMap_Chatty1.0.pkl'),'wb') as f:

    pickle.dump(filteredToUnfiltedPositionMap, f, protocol=2)
