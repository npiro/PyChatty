# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 11:58:06 2016

@author: Jean-Christophe
"""

import message_db.message_db as mdb
import numpy as np

from chatbot.ChattyV2 import ChattyV2

class ChattyV3(ChattyV2):
    def __init__(self, folder = 'C:\\Users\\Client\\Dropbox\\S2DS - M&S\\Data', 
                 filenameClient = '05_clientMessagesFilteredfastText.txt', 
                 filenameAgent = '05_agentMessagesFilteredfastText.txt',
                 filenameMessageDF = 'MessagesDF.pkl',
                 filenameW2V = 'W2V_Chatty2_0.mdl', 
                 filenameSearchTree = 'SearchTree_Chatty2_0.pkl',
                 filenameMessageMap = 'Filtered2UnfilteredMessageMap_Chatty2_0.pkl',
                 filenameMessageInfoTable = 'client_agent_summary3.csv',
                 filenameMatrix = 'SentenceMatrix_Chatty2_0.pkl',
                 load_from_file = True, levelFactor = 0.35):
                     
        # Init bot
                     
        # Init V2 stuff
        super(ChattyV3,self).__init__(folder, filenameClient, filenameAgent,
                 filenameMessageDF, filenameW2V, filenameSearchTree,
                 filenameMessageMap, filenameMessageInfoTable, filenameMatrix,
                 load_from_file = True, levelFactor = 0.35)
                 

        # Init V3 answer finder
        self.answerObject = AnswerFinder(self.sentenceMatrixExtended)
        
        self.name = 'Chatty 3.0'

    def resetChat(self):
        self.answerObject.refreshConversation()
    

    
    def getReplyFromChatBot(self, input_message):
        try:
            print(input_message)
            # Apply preprocessing to input query
            input_message = self.cleaner.clean_input(input_message)
            print(input_message)
            # Vectorize message
            vect = self.vectorizer.get_sentence_vector(input_message)
            vect= vect/self.MaximumVectorLength
            #print(vect)
            level = self.answerObject.getConversationStatus()
            vect = np.hstack((vect,level*self.levelFactor))
            print(vect)

            # Find nearest message in client corpus
            ind = self.answerObject.getNextAnswer(vect)
            print(ind)
            #print(self.answerObject.getDistances())
            # Unmap to account for filter-removed messages
            ind = self.FilteredToUnfilteredMessageMap[ind]
            print(ind)

            # Get closest client and agent message
            df = self.MessagesDataFrame
            ClientMessage = mdb.get_client_message_from_dataframe(df, ind)
            AgentMessage = mdb.get_agent_message_from_dataframe(df, ind)

            return (AgentMessage, ClientMessage)

        except Exception as inst:
            print type(inst)     # the exception instance
            print inst.args      # arguments stored in .args
            print inst           # __str__ allows args to be printed directly
            
    def changeLevelFactor(self, levelFactor):
        self.levelFactor = levelFactor
        # Extend sentence matrix with one dimention equal to level
        self.sentenceMatrixExtended=np.hstack((self.sentenceMatrixNormalized,
                                              self.sentenceLevelNoNans[self.correspondenceVector]*self.levelFactor))
                                              
        self.answerObject = AnswerFinder(self.sentenceMatrixExtended)
        
        print('Chenged level to'+str(self.levelFactor))
        
        
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
        distances = self.calculateDistances(vectorisedUserQuery)
        indexNewResponse = self.findBestAnswer(distances)
        self.updateListAnswer(indexNewResponse)

        return indexNewResponse

    def calculateDistances(self, vectorisedUserQuery):
        """
        Function to calculate the distance as defined for Chatty 3.0
        It returns the list of distances for every sentence of sentenceMatrix

        """
        sizeVecSpace = len(self.matrixSentence[:,0])

        dist = np.zeros(sizeVecSpace)
        weightCoeff = self.calculateWeightCoeff()

        # loop over all past responses, to calculate the calculate the sum of the corresponding distances for each vector.
        for responseIndexes in range(self.responseNumber) :
            #The point in respect to which the distance is calculated at each iteration corresponds to the vector
            #of the given answer at that stage.
            centre = self.matrixSentence[self.pastResponses[responseIndexes].astype(int),:]

            # Loop over each lines of matrixSentence
            for lineNumber in range(sizeVecSpace):
                #the distance is a dot product
                dist[lineNumber] = np.vdot((self.matrixSentence[lineNumber - self.responseNumber + responseIndexes,:]-centre),(self.matrixSentence[lineNumber - self.responseNumber +  responseIndexes,:]-centre) ) * weightCoeff[responseIndexes] + dist[lineNumber]

        #We also include the distance with respect to the current input query.
        for lineNumber in range(sizeVecSpace):
            centre = vectorisedUserQuery
            dist[lineNumber] = np.vdot((self.matrixSentence[lineNumber,:]-centre),(self.matrixSentence[lineNumber,:]-centre) ) * weightCoeff[self.responseNumber] + dist[lineNumber]

        return dist

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
        cutoff = 1.

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

    def getDistances(self):

        return self.distances
