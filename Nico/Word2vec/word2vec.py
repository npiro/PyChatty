
# coding: utf-8

# # Training word2vec

# First we load text data

# In[1]:

import os.path

import gensim, logging

import codecs

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

folder = '/Users/piromast/Dropbox/S2DS - M&S/Data/'
agentFileName = os.path.join(folder, '04_agentMessagesFiltered.txt')
clientFileName = os.path.join(folder, '04_clientMessagesFiltered.txt')
clientAgentFileName = os.path.join(folder, '04_clientAgentMessagesFiltered.txt')



# Create generator class to load data line by line

# In[2]:

class SentencesGenerator(object):
    def __init__(self, filename):
        self.filename = filename
 
    def __iter__(self):
        for line in codecs.open(self.filename,'r','utf-8'):
            yield line.split()


# Create iterator object for agent and client files

# In[3]:

clientSentencesGenerator = SentencesGenerator(clientFileName)
agentSentencesGenerator = SentencesGenerator(agentFileName)
clientAgentSentencesGenerator = SentencesGenerator(clientAgentFileName)
#sentences=list(clientSentencesGenerator)


# In[4]:
# Create a dictonary
dictionary = gensim.corpora.Dictionary(clientSentencesGenerator)
dictionary.save(os.path.join(folder,'ClientDictionary.dict'))

# In[5]:
# Train model with client messages
modelClient = gensim.models.Word2Vec(clientSentencesGenerator, size=100, window=5, min_count=3, workers=4)


# In[6]:
# Train model with agent messages
modelAgent = gensim.models.Word2Vec(agentSentencesGenerator, size=100, window=5, min_count=3, workers=4)

# In[7]:
# Train model with both agent and client meesages from joint messages corpus
modelBothTogether = gensim.models.Word2Vec(clientAgentSentencesGenerator, size=100, window=5, min_count=3, workers=4)

# In[8]:
# Train model with both agent and client meesages from each seperate corpus once at a time
modelBothSeperate = gensim.models.Word2Vec(clientSentencesGenerator, size=100, window=5, min_count=3, workers=4)

modelBothSeperate.train(agentSentencesGenerator)


# In[9]:

# Save models

modelClient.save(os.path.join(folder,'clientW2V_Model'))
modelAgent.save(os.path.join(folder,'clientW2V_Agent'))
modelBothTogether.save(os.path.join(folder,'clientW2V_BothTogether'))
modelBothSeperate.save(os.path.join(folder,'clientW2V_BothSeperate'))


# In[10]:
# Try models
print(modelClient.most_similar(positive=['girl'])[0:2])
print(modelAgent.most_similar(positive=['girl'])[0:2])
print(modelBothTogether.most_similar(positive=['girl'])[0:2])
print(modelBothSeperate.most_similar(positive=['girl'])[0:2])
