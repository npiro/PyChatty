# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 09:35:59 2016

@author: piromast
"""
from evaluation.evaluation import ChatEvaluator
import os.path
import numpy as np
from matplotlib.pyplot import *
import seaborn

folder = '/Users/piromast/Dropbox/S2DS - M&S/Data'
folderSave = '/Users/piromast/Dropbox/S2DS - M&S/Data/'
filenameAgent = '04_agentMessagesFilteredTrigramLemmatized.txt'
filenameAgent = os.path.join(folder,filenameAgent)
filenameClient = '04_agentMessagesFilteredTrigramLemmatized.txt'
filenameClient = os.path.join(folder,filenameClient)
evaluator = ChatEvaluator(filenameClient, filenameAgent)

Cos_simils, Input_dists, Output_dists = [], [], []

sizes = range(50,200,25)
for size in sizes:
    cosSim, distIn, distOut = evaluator.test(size = size, num_splits = 5)
    Cos_simils.append(cosSim)
    Input_dists.append(distIn)
    Output_dists.append(distOut)

np.savetxt(os.path.join(folderSave,'Cos_similarity.txt'),Cos_simils)
np.savetxt(os.path.join(folderSave,'Input_dists.txt'),Input_dists)
np.savetxt(os.path.join(folderSave,'Output_dists.txt'),Output_dists)

# In[2]:
CosSimMean = np.mean(Cos_simils,1)
pl=plot(list(sizes),CosSimMean)
xlabel('Vector dimensions')
ylabel('Total cosine similarity')
