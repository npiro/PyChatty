# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 12:21:42 2016

@author: piromast
"""

from vectorisations.vectorization import TfIdf
import os

folder = '/Users/piromast/Dropbox/S2DS - M&S/Data'
filename = '04_clientMessagesFilteredTrigramLemmatized.txt'

fn = os.path.join(folder,filename)

tfidf = TfIdf(fn)


# In[1]:
[tfidf.model[w] ]

# In[2]:
