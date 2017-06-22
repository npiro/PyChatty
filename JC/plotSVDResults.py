# -*- coding: utf-8 -*-
"""
Created on Fri Aug 05 16:19:15 2016

@author: Jean-Christophe
"""
import matplotlib.pyplot as plt
#import seaborn as sb
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import gensim, logging

dimension=50
dimensionToPlot = 3

def plot_svd(file_name):
    svFile=open(file_name,'r')
    sValues = np.empty([dimension])
    xRange = np.empty([dimension])
    for i in range(dimension):
        sValues[i]  = svFile.readline()
        xRange[i] = i 
    plt.bar(xRange,sValues)
    plt.ylabel("Singular value")
    plt.xlabel("Singular values order")

plt.figure()
plot_svd('svTrigram')
plt.figure()
plot_svd('svAgent')


urFile=open('UrTrigram','r')
#testUr=np.genfromtxt(StringIO(urFile.readlines()))


def plot_coocurrence_svd2D(file_name,Ur_filename,words_number=100):
    urFile=open(Ur_filename,'r')
    
    words = list()
    vectors = np.empty([words_number, dimensionToPlot])
    for i in range(words_number):
        print i
        L1list = urFile.readline()
        if dimension == dimensionToPlot:
            [frequency,word, val_1, val_2] = L1list.split(' ')
        else:
            [frequency,word, val_1, val_2, dump] = L1list.split(' ',4) 
        print [frequency,word, val_1, val_2]
        words.append(word.decode('utf8') )
        vectors[i,0] = val_1
        vectors[i,1] = val_2
    #print words
    #print vectors
    plt.scatter(vectors[:,0], vectors[:,1])
    for word, x, y in zip(words, vectors[:,0], vectors[:,1]):
        plt.annotate(word, (x, y), size=12)
    
    plt.title(file_name)
    plt.autoscale(True, 'both', True)

def plot_coocurrence_svd3D(file_name,Ur_filename,words_number=100):
    urFile=open(Ur_filename,'r')

    words = list()
    vectors = np.empty([words_number, dimensionToPlot])
    for i in range(words_number):
        L1list = urFile.readline()
        if dimension == dimensionToPlot:
            [frequency,word, val_1, val_2, val_3] = L1list.split(' ')
        else:
            [frequency,word, val_1, val_2, val_3, dump] = L1list.split(' ',5)
        words.append(word.decode('utf8') )
        vectors[i,0] = val_1
        vectors[i,1] = val_2
        vectors[i,2] = val_3
    #print words
    #print vectors
        
    plt.scatter(vectors[:,0], vectors[:,1], 20, vectors[:,2])
    for word, x, y in zip(words, vectors[:,0], vectors[:,1]):
        plt.annotate(word, (x, y), size=12)
    
    plt.title(file_name)
    plt.autoscale(True, 'both', True)
    
def plot_coocurrence_svd4D(file_name,Ur_filename,words_number=100):
    urFile=open(Ur_filename,'r')

    words = list()
    vectors = np.empty([words_number, dimensionToPlot])
    for i in range(words_number):
        L1list = urFile.readline()
        if dimension == dimensionToPlot:
            [frequency,word, val_1, val_2, val_3, val_4] = L1list.split(' ')
        else:
            [frequency,word, val_1, val_2, val_3, val_4, dump] = L1list.split(' ',6)
        words.append(word.decode('utf8') )
        vectors[i,0] = val_1
        vectors[i,1] = val_2
        vectors[i,2] = val_3
        vectors[i,3] = val_4
    #print words
    #print vectors
        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vectors[:,0], vectors[:,1], vectors[:,3], 'z', 20, vectors[:,2])


    for word, x, y, z in zip(words, vectors[:,0], vectors[:,1], vectors[:,3]):
        ax.text(x, y, z, word)
    
    plt.title(file_name)
    plt.autoscale(True, 'both', True)

def plot_coocurrence_svd5D(file_name,Ur_filename,words_number=100):
    urFile=open(Ur_filename,'r')

    words = list()
    vectors = np.empty([words_number, dimensionToPlot])
    for i in range(words_number):
        L1list = urFile.readline()
        if dimension == dimensionToPlot:
            [frequency,word, val_1, val_2, val_3, val_4, val_5] = L1list.split(' ')
        else:
            [frequency,word, val_1, val_2, val_3, val_4, val_5, dump] = L1list.split(' ',7)
        words.append(word.decode('utf8') )
        vectors[i,0] = val_1
        vectors[i,1] = val_2
        vectors[i,2] = val_3
        vectors[i,3] = val_4
        vectors[i,4] = val_5

    #print words
    #print vectors
        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vectors[:,0], vectors[:,1], vectors[:,3], 'z', vectors[:,4]*20000, vectors[:,2])    

    for word, x, y, z in zip(words, vectors[:,0], vectors[:,1], vectors[:,3]):
        ax.text(x, y, z, word)
    plt.title(file_name)    
    plt.autoscale(True, 'both', True)

plt.figure()
if dimensionToPlot == 2:
    plot_coocurrence_svd2D('Client word space, d=2','UrTrigram',words_number=100)
    plot_coocurrence_svd2D('Agent word space, d=2','UrAgent',words_number=100)
elif dimensionToPlot == 3:
    plt.figure()
    plot_coocurrence_svd3D('Client word space, d=3','UrTrigram',words_number=50)
    plt.xlim(0.996, 1.004)
    plt.ylim(-0.01, 0.005)
    plt.figure()
    plot_coocurrence_svd3D('Agent word space, d=3','UrAgent',words_number=50)
      #  plt.xlim(0.996, 1.004)
    plt.ylim(-0.006, 0)
elif dimensionToPlot == 4:
    plot_coocurrence_svd4D('Client word space, d=4','UrTrigram',words_number=50)
    plot_coocurrence_svd4D('Agent word space, d=4','UrAgent',words_number=50)
elif dimensionToPlot == 5:
    plot_coocurrence_svd5D('Client Word space, d=5','UrTrigram',words_number=100)
    plot_coocurrence_svd5D('Agent Word space, d=5','UrAgent',words_number=100)
# plt.ylim(ymin, ymax)
    
    

# Plotting Word2Vec results


def plot_Word2Vec5D(file_name,Ur_filename,words_number=100):
    ModelFileName ='clientW2V_BothSeperate'
    model = gensim.models.Word2Vec.load(ModelFileName)
    urFile=open(Ur_filename,'r')
    plt.figure()

    words = list()
    urFile.readline()
    vectors = np.empty([words_number, 5])
    for i in range(words_number):
        L1list = urFile.readline()
        [frequency, word, dump] = L1list.split(' ',2)
        words.append(word.decode('utf8') )
        vectors[i,0] = model[word][0]
        vectors[i,1] = model[word][1]
        vectors[i,2] = model[word][2]
        vectors[i,3] = model[word][3]
        vectors[i,4] = model[word][4]

    #print words
    #print vectors
        
    fig = plt.figure()
    plt.xlim(0, 1)
    xmin=-2
    xmax=2
    ymin=-2
    ymax=2
    
    
    ax = fig.add_subplot(111, projection='3d')
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    ax.scatter(vectors[:,0], vectors[:,1], vectors[:,3], 'z', vectors[:,4]*50, vectors[:,2])
    for word, x, y, z in zip(words, vectors[:,0], vectors[:,1], vectors[:,3]):
        if x < xmax and x > xmin and y < ymax and y > ymin:
            ax.text(x, y, z, word)
    plt.title(file_name)    


    #plt.autoscale(True, 'both', True)
    
plot_Word2Vec5D('Word space Word2Vec, d=5','Ur',words_number=100)
