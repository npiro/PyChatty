
# In[1]:
# Import stuff
import os.path
import gensim, logging
import codecs
import numpy as np

# In[2]:
# Load model
ModelFileName ='clientW2V_BothSeperate'
model = gensim.models.Word2Vec.load(ModelFileName)

print model['cost']

