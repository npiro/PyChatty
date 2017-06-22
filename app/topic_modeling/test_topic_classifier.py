# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 14:34:16 2016

@author: Client
"""
from topic_modeling.topic_modeling import LDATopicClassifier


folderNicoMac = '/Users/piromast/Dropbox/S2DS - M&S/Data'
folder = folderNicoMac

topicClassifier = LDATopicClassifier(folder=folder)

# In[1]:
import matplotlib.pyplot as plt
text = u'Sparks sparks sparks order order hello'
topics = topicClassifier.getMSTopicDistri([text])


# In[1]:
f = plt.figure
a = plt.axes()
topicClassifier.getMSTopicHistogram([text], a)