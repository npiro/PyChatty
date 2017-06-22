
# In[1]:
# Import stuff
import os.path
import gensim, logging
import codecs
import numpy as np

# In[2]:
# Load model
folder = '/Users/piromast/Dropbox/S2DS - M&S/Data/'
ModelFileName ='clientW2V_BothSeperate'
model = gensim.models.Word2Vec.load(ModelFileName)

