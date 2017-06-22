# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 15:50:28 2016

@author: Jean-Christophe

"""
import numpy as np

class AnswerFinder :
    """
    Class to search for best answer for Chatty 3.0
    It will take vectorised sentences from the user, and return the index of the best
    match for the answer.
    """
    
    def __init__(self, matrixSentence, pastResponses = [], responseNumber = 0):
        """
        Constructor, taking into input
        - matrixSentence: the matrix with the list of vectorised corpus (as given by Word2Vec)
        - pastReponses: the list which will keep track of the indexes of the past answers given to the client
        By default, it's empty to start with.
        - responseNumber = an integer to keep track of how many answers have been given to the client already.
        By default, it'll be initialised at 0.
        """

        self.matrixSentence = matrixSentence  
        self.pastResponses = pastResponses
        self.responseNumber = responseNumber
        
    def getNextAnswer(self, vectorisedUserQuery):
        """
        Function to get the index of the "best" next answer. It simply calls the intermediate functions,
        and return the index of the best answer (corresponding to the right line of matrixSentence)
        """
        distances =self.calculateDistances(vectorisedUserQuery)
        indexNewResponse = self.findBestAnswer(distances)
        self.updateListAnswer(indexNewResponse)
        
        return indexNewResponse

    def calculateDistances(self, vectorisedUserQuery):
        """
        Function to calculate the distance as defined for Chatty 3.0
        It returns the list of distances for every sentence of sentenceMatrix
        
        """
        sizeVecSpace = len(self.matrixSentence[:,0])
        
        distances = np.zeros(sizeVecSpace)
        weightCoeff = self.calculateWeightCoeff()
        
        # loop over all past responses, to calculate the calculate the sum of the corresponding distances for each vector.
        for responseIndexes in range(self.responseNumber) :            
            #The point in respect to which the distance is calculated at each iteration corresponds to the vector 
            #of the given answer at that stage.
            centre = self.matrixSentence[self.pastResponses[responseIndexes].astype(int),:]
            
            # Loop over each lines of matrixSentence
            for lineNumber in range(sizeVecSpace):
                #the distance is a dot product
                distances[lineNumber] = np.vdot((self.matrixSentence[lineNumber]-centre),(self.matrixSentence[lineNumber]-centre) ) * weightCoeff[responseIndexes] + distances[lineNumber]
        
        #We also include the distance with respect to the current input query.
        for lineNumber in range(sizeVecSpace):
            centre = vectorisedUserQuery
            distances[lineNumber] = np.vdot((self.matrixSentence[lineNumber]-centre),(self.matrixSentence[lineNumber]-centre) ) * weightCoeff[self.responseNumber] + distances[lineNumber]
        
        return distances
   
    def findBestAnswer(self, distances):
        """
        Function to return the index of which line of sentenceMatrix (=sentence) returns 
        the minimal distance (it works even if NaNs are in the matrix)
        """

        indexNewResponse = np.nanargmin(distances)
        
        return indexNewResponse
    
    def updateListAnswer(self, indexNewResponse):
        """
        Function to update the list of past responses and the number of responses already given.
        """
        
        self.pastResponses = np.hstack((self.pastResponses, indexNewResponse))
        self.responseNumber = self.responseNumber + 1
        
    def calculateWeightCoeff(self):
        """
        Function to calculate a weight coefficient in the sum to give more or less weight to the memory.
        Right now, it's simply the inverse function. Playing with the cutoff threshold could be useful.        
        """
        cutoff = 0.33
        
        weightCoeff= np.empty(self.responseNumber + 1)
        
        for responseIndexes in range(self.responseNumber + 1):
            weightCoeff[responseIndexes] = 1/(cutoff * (self.responseNumber + 1 - responseIndexes) )
        print(weightCoeff)
        return weightCoeff
    
    def refreshConversation(self):
        """
        Function to restart a conversation from scratch, i.e. re-intialise pastResponses and responseNumber
        """
        self.pastResponses = []
        self.responseNumber = 0
    
    def getConversationStatus(self):
        """
        Function to return how many answers have been given already.
        """
        return self.responseNumber
        
        