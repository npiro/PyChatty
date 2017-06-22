# -*- coding: utf-8 -*-
"""
Created on Fri Aug 05 16:19:15 2016

@author: Jean-Christophe
""" 
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import pandas as pd
from StringIO import StringIO

svFile=open('/home/walter/Dropbox/S2DS - M&S/DevCode/MS/Walter/cca-master/output/agentMessages.cutoff1.window10.m2.kappa2.out/Ur','r')


#words = ['queen', 'book', 'king', 'magazine', 'car', 'bike']
#vectors = np.array([[0.1,   0.3],  # queen
                    #[-0.5, -0.1],  # book
                    #[0.2,   0.2],  # king
                    #[-0.3, -0.2],  # magazine
                    #[-0.5,  0.4],  # car
                    #[-0.45, 0.3]]) # bike

words_number = 67
#words_number = 68
words = list()
vectors = np.empty([words_number, 2])

for i in range(words_number):
    print i
    L1list = svFile.readline()
    [frequency,word, val_1, val_2] = L1list.split()
    print [frequency,word, val_1, val_2]
    words.append(word)
    vectors[i,0] = val_1
    vectors[i,1] = val_2
#print words
#print vectors

plt.figure()
plt.scatter(vectors[:,0], vectors[:,1])
#plt.show()
for word, x, y in zip(words, vectors[:,0], vectors[:,1]):
    sb.plt.annotate(word, (x, y), size=12)

plt.show()




#plt.figure()
#plt.bar([1, 2, 3, 4, 5],[0.1, 0.4, 0.01, 0.2, 0.05])

##s.plot(kind='bar')
##sb.plt.ylabel("$P(w|Cinderella)$")

#urFile=open('Ur','r')
#testUr=np.genfromtxt(StringIO(urFile.readlines()))








