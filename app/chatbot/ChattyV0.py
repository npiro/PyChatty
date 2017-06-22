# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 23:29:55 2016

@author: N.Piro
"""

class ChattyV0(object):

    def __init__(self):

        # Init properties
        self.name = 'Chatty 0.0'

        # Message number
        self.msg_num = 0
        self.messageList = ['Hello, how can I help you?','Can you tell me more about it?',
                      "Not sure of what the problem is exactly. Can you please be more specific?",
                      "Ok, I'll pass you with a specialized agent"]
                      
    def getReplyFromChatBot(self, input_message):
        if self.msg_num > len(self.messageList)-1: self.msg_num = len(self.messageList)-1
        output = self.messageList[self.msg_num]
        self.msg_num += 1
        return (output, ' ')


    def save(self, folder = None, filenameW2V = None, filenameMatrix = None,
             filenameSearchTree = None, filenameMessageMap = None):
        return

    def load(self, folder = None, filenameW2V = None, filenameMatrix = None,
             filenameSearchTree = None, filenameMessageMap = None):
        return

    def resetChat(self):
        return

    def changeLevelFactor(self, levelFactor):
        return
