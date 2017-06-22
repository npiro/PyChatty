# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 16:45:10 2016

Tools to vectorize words, sentences, phrases and documents.

class KMeanClusterFinder(): iterator class to run through text line by line

Example code:
    (add app folder to PYTHONPATH)
    import clustering
    # create clustering Object, specifying the matrix to start with, the K-Means method to use, 
    # and the number of (maximum) clusters
    testClusters= clustering.KMeanClusterFinder(matrix,"MiniBatchKMeans", 12) 
    
    # finding the optimum of clusters, based on silhouette score, and returns centroids, labels and silhouette score of the optimum    
    centers, labels, silhouetteScore  = testClusters.find_optimised_number_clusters()
    # write a random selection of the cluster text content on a file
    testClusters.write_clusters_file("03_clientMessagesMatch04.txt", 20)

@author: Jean-Christophe
"""

from sklearn.cluster import KMeans, MiniBatchKMeans
from time import time
# notice this import, you might have to include it in your PATH
import filteringNaNs
import numpy as np
import os.path
import codecs
from sklearn import metrics
import random
import matplotlib.pyplot as plt

class KMeanClusterFinder(object):
    """
    Class to create an object associated with the clustering oparation, using K-Means algorithm

    """
    
    def __init__(self, matrix, method, numberOfClusters):
        """
             Class constructor. It takes as input:
             - matrix= the matrix to find clusters from, directly obtained from Word2Vec (or Doc2Vec, etc). 
             The points to cluster should be the rows of the matrix, the column are their spatial coordinates. 
             The matrix can contain NaNs (actually it's better not to filter them to get the clustering, for self-consistency of the pipeline). 
             - method to use. So far, choice "MiniBatchKMeans" or "KMeans". MiniBatchKMeans is in theory faster but less accurate than KMeans.
             (KMeans is parallelised though but MiniBatchKMeans isn't...)
             - number of Clusters: either the number of clusters you want to find, or the maximum number of clusters if you want to 
             find a "clustering optimum".
        """
        self.matrixToCluster = matrix
        self.method = method
        self.numberOfClusters = numberOfClusters
        # other class variables we initalise to empty for subsequent functions
        self.centers = np.empty( numberOfClusters )
        self.labels = np.empty( len( matrix[:,1] ) )
         
         
    def find_clusters(self):
        """
           Function to find the clusters by applying the desired K-means algorithm. 
           It will search for a number of clusters: self.numberOfClusters.
           This function calls the function find_clusters_routine() a certain number of times
           (defined by numberOfTries), to avoid finding a local minima (K-means starts on a random configuration which 
           can lead to being stuck in a local minima)
          
           Returns as outputs
            - the coordinates of the cluster centroids (matrix of size numberOfClusters * dimensionOfSpace)
            - the obtained labels = which cluster does a point belong to. (vector of size: numberOfPoints)
            Note that these labels can contain NaNs (if the corresponding row in matrixToCluster is NaNs).
            - the silhouette score of the clustering
            
            VERY IMPORTANT NOTE: right now the clustering is done on the input matrix self.matrixToCluster as is. 
            Some people claim that a dimensionality reduction or re-normalisation could improve clustering,
            but this is not currently implemented as it needs to be tested.
        """
        
        #We apply the filtering function to get rid of NaNs line in self.matrixToCluster (otherwise K-means will crash)
        matrixNoNaNs, correspondanceMatrix  = self.apply_NaN_filtering()

        numberOfTries = 10
        
        bestMetricsScore= -1
        
        for i in range(numberOfTries):
            centersTemp, labelsTemp, silhouetteScore =  self.find_clusters_routine(matrixNoNaNs, correspondanceMatrix)

            if silhouetteScore > bestMetricsScore:
                bestLabels = labelsTemp
                bestCenters = centersTemp
                bestMetricsScore = silhouetteScore

        self.centers = bestCenters
        #We convert back the labels obtained by K-means to the corresponding points in self.matrixToCluster, before the NaNs were removed.
        self.make_NaN_filtered_labels_correspond_to_orginial_matrix(matrixNoNaNs, correspondanceMatrix, bestLabels)
        
        return bestCenters, self.labels, bestMetricsScore
        
    def find_clusters_routine(self,  matrixNoNaNs, correspondanceMatrix):
        """
           Function to actually apply K-means, called by find_clusters(self). 
        """
        
        verbose    = 'no'
        if self.numberOfClusters > 1:
                
            if self.method == "MiniBatchKMeans":
                km = MiniBatchKMeans(n_clusters=self.numberOfClusters, init='k-means++', n_init=1,
                                     init_size=1000, batch_size=1000, verbose=verbose)
            elif self.method == "KMeans":
                km = KMeans(n_clusters=self.numberOfClusters, init='k-means++', max_iter=100, n_init=1,
                            verbose=verbose, n_jobs = -2)
            else:
                raise ValueError('Unknown task_id ' + self.method)
        else:
            raise ValueError('Non-positive number of clusters: ' + self.numberOfClusters )
    
    
        print("Clustering data with %s" % km)
        t0 = time()
        km.fit(matrixNoNaNs)
        print("done in %0.3fs" % (time() - t0))
        silhouetteScore = metrics.silhouette_score(matrixNoNaNs, km.labels_, sample_size=2000)
        print("Silhouette Coefficient: %0.3f"% silhouetteScore)
    
        return km.cluster_centers_ , km.labels_ , silhouetteScore
        
    def find_optimised_number_clusters(self, plottingEnabled):
        """
        Function to return optimum the number of clusters, i.e. the one which makes the silhouette coefficient closest to 1.
        It will try all possibilities, from 2 clusters to numberOfClusters.
        The input plottingEnabled (needs to be a Boolean) enables to display (or not) a figure showing the 
        variaton of the silhouette coefficient with cluster number.
        """
        silhouetteScoreList= np.empty( self.numberOfClusters -1)
        
        if self.numberOfClusters > 1:

            bestMetricsScore=-1
            bestNumberClusters=-1

            for i in range(2, self.numberOfClusters+1):
                self.numberOfClusters=i
                centers, labels, silhouetteScoreList[i-2] = self.find_clusters()
                
                if silhouetteScoreList[i-2] > bestMetricsScore:
                    bestNumberClusters = i
                    bestMetricsScore = silhouetteScoreList[i-2]
        else:
            raise ValueError('Non-positive number of clusters: ' + self.numberOfClusters )
        if plottingEnabled:
           self.plotOptimisedClusters(silhouetteScoreList) 
        
        
        print("The optimal number of clusters is " + str(bestNumberClusters))
        self.numberOfClusters = bestNumberClusters
        centers, labels, silhouetteScore  = self.find_clusters()
       
        return centers, labels, silhouetteScore 

 
          
    def apply_NaN_filtering(self):
        """
        Function to remove NaNs in the matrixToCluster
        """
        
        matrixNoNaNs, correspondanceMatrix = filteringNaNs.get_matrix_without_NaNs(self.matrixToCluster)
        return matrixNoNaNs, correspondanceMatrix 
                
    def make_NaN_filtered_labels_correspond_to_orginial_matrix(self, matrixNoNaNs, correspondanceMatrix, labels):
        """
        Function to make the labels obtained from the matrix where the NaNs have been removed 
        into a new list of labels that correspond to the initial matrix.
        """

        lineList = np.empty( len( matrixNoNaNs[:,1] ) )
        correspondingLabels = np.empty( len( self.matrixToCluster[:,1] ) )
            
        for j in range(len(matrixNoNaNs[:,1])):
                lineList[j] = j        

        for i in range(self.numberOfClusters):

            selectLabel = ( labels == i )
            selectLinesLabels = lineList[ selectLabel ].astype(int)
            correspondingLabels[correspondanceMatrix[selectLinesLabels.astype(int)]]= i
            
        self.labels = correspondingLabels
        

    def write_clusters_file(self, fileName, numberOfPointPerCluster):
        """
        Function to write down a random number of points per cluster in a file, Clusters.txt .
        The input is:
        - fileName: the name of the file where the original text is. IT NEEDS TO correspond line by line to the 
        self.matrixToCluster.
        - numberOfPointPerCluster: the number of points per cluster to write down in the file.
        
        """
        
        pointSelected = np.empty( [numberOfPointPerCluster , self.numberOfClusters] )
    
        filename = os.path.join("..","..","Data",fileName)
        
        with codecs.open(filename,'r','utf-8') as f:
            clientMessages = f.readlines()
        
        lineList = np.empty( len( self.matrixToCluster[:,1] ) )
            
        for j in range(len(self.matrixToCluster [:,1])):
            lineList[j] = j
    
    
        for i in range(self.numberOfClusters):
        
            selectLabel = ( self.labels == i )
            selectLinesLabels = lineList[ selectLabel ].astype(int)
            
            for k in range(numberOfPointPerCluster):
                pointSelected[k,i] = random.choice(selectLinesLabels) 
        
        fileClusters = codecs.open('Clusters.txt','w','utf-8')

        for i in range(self.numberOfClusters ):
            fileClusters.write('Cluster number '+str(i)+'\n')
            for j in range(numberOfPointPerCluster):
                fileClusters.write(clientMessages[pointSelected[j,i].astype(int)])
            fileClusters.write('\n')
            fileClusters.write('*********************\n')
            fileClusters.write('\n')


        fileClusters.close()
    
    def plot_clusters():
        """
        Function to plot the obtained clusters. TO DO.
        """
        
        return
        
    def plotOptimisedClusters(self, silhouetteScoreList):
        
        x=range(2, len(silhouetteScoreList)+2)
        plt.figure
        plt.plot(x, silhouetteScoreList)
        plt.title("Variation of the silhouette score with cluster numbers")
        plt.ylabel("Silhouette score")
        plt.xlabel("Cluster numbers")
        plt.autoscale(True, 'both', True)
        return
    
