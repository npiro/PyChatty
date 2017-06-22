
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 19:49:30 2016

@author: N. Piro
"""

from PyQt4 import QtGui, QtCore, uic
from MplFigure.MplFigure import MplFigure
import pandas as pd
import numpy as np
import os
import message_db.message_db as mdb
from chatbot.ChattyV0 import ChattyV0
from chatbot.ChattyV1 import ChattyV1
from chatbot.ChattyV2 import ChattyV2
from chatbot.ChattyV3 import ChattyV3
from topic_modeling import topic_modeling
from sentiment.sentiment import Sentiment
from matplotlib.pylab import plt
from preprocessing.clean_input import InputCleaner

Ui_MainWindow, QMainWindow = uic.loadUiType("ChatGui.ui")




folderJC = 'D:\\Dropbox\\S2DS - M&S\\Data'
folderNico = 'C:\\Users\\Client\\Dropbox\\S2DS - M&S\\Data'
folderNicoMac = '/Users/piromast/Dropbox/S2DS - M&S/Data'
folder = folderNicoMac


class ChatGuiMainWindow(QtGui.QMainWindow,Ui_MainWindow):
    def __init__(self, folder = folder, db_filename = 'MessagesDB.db'):

        super(ChatGuiMainWindow, self).__init__()
        self.setupUi(self)

        # customize the UI
        self.initUI()

        # init class data
        self.initData(folder, db_filename)

        # connect slots
        self.connectSlots()

        # init MPL widget
        self.initMplWidget()

        self.cleaner = InputCleaner(folder)

    def __del__(self):
        mdb.disconnect_from_db(self.db)

    def initUI(self):

        # mpl figure
        self.PlotFigure = MplFigure(self)
#        #self.gridLayout.PlotLayout.addWidget(self.main_figure.toolbar)
        self.PlotLayout.addWidget(self.PlotFigure.canvas)


        #self.setGeometry(300, 300, 350, 300)
        self.setWindowTitle('Chatty V2.0')
        self.show()



    def initData(self, folder, db_filename):
        self.name = 'nico'
        self.chatText.setText('')
        self.topicClassifier = topic_modeling.LDATopicClassifier(folder=folder)
        self.sentimentAnalyzer = Sentiment()
        self.loadTopicTableData()

        # Initialize chat message history
        self.clearChatHistory()

        # Open message database
        fn = os.path.join(folder, db_filename)
        self.db = mdb.connect_to_db(fn)

        # Init user name
        self.userName = 'M&S team'

    def connectSlots(self):
        self.inputQueryText.returnPressed.connect(self.inputQueryTextReturn)
        self.resetButton.released.connect(self.resetButtonPressed)
        self.enterButton.released.connect(self.enterButtonPressed)
        self.levelFactorText.textChanged.connect(self.levelFactorChanged)
        self.saveModelButton.released.connect(self.saveModelButtonPressed)
        self.versionCombo.currentIndexChanged.connect(self.versionChanged)
        self.topicTable.cellClicked.connect(self.tableCellClicked)

    def initMplWidget(self):
        """creates initial matplotlib plots in the main window and keeps
        references for further use"""
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(3, 1)

        # top plot
        self.ax_top = self.PlotFigure.figure.add_subplot(gs[:2,0])


        #self.ax_top.set_ylim(-32768, 32768)
        #self.ax_top.set_xlim(0, 3)
        #self.ax_top.set_xlabel(u'time (ms)', fontsize=6)

        # bottom plot
        self.ax_bottom = self.PlotFigure.figure.add_subplot(gs[2,0])
        #self.ax_bottom.set_ylim(0, 1)
        #self.ax_bottom.set_xlim(0, 1)
        #self.ax_bottom.set_xlabel(u'frequency (Hz)', fontsize=6)

        # line objects
        #self.line_top = self.ax_top.hist(range(10), range(10))

        #self.line_bottom = self.ax_bottom.hist(range(10), range(10))


    def inputQueryTextReturn(self):
        # Get the input text and clear text box
        user_text = str(self.inputQueryText.text())

        self.inputQueryText.clear()

        # Append message to chat text box
        #self.chatText.append('<'+self.name+'> '+user_text)
        self.chatText.append('<font color="blue">&lt;'+self.userName+'&gt; '+user_text+'</font>')

        self.chatText.show()

        # Get agent reply and nearest input
        agent_reply, closest_user_input = self.chatBot.getReplyFromChatBot(user_text)


        # Output reply to chat box
        #self.chatText.append('<'+self.chatBot.name+'> '+agent_reply)
        self.chatText.append('<font color="green">&lt;'+self.chatBot.name+'&gt; '+user_text+'</font>')
        self.chatText.append('')
        self.chatText.show()
        self.nearestInputText.setText(closest_user_input)
        self.nearestInputText.show()
        self.updateChatData(user_text, agent_reply)
        self.classifyTopic()
        self.analyzeSentiment()
        print(closest_user_input)

    def updateChatData(self, user_text, agent_text):
        """
        Update chat history, do topic classification if required
        Add text to sentiment analyzer
        """
        self.chatHistoryUser.append(user_text)
        self.chatHistoryAgent.append(agent_text)
        self.sentimentAnalyzer.add_client_sentence(user_text)
        self.sentimentAnalyzer.add_agent_sentence(agent_text)

    def classifyTopic(self):

        if self.classifyTopicCheck.isChecked():
            self.ax_top.cla()
            topicClass = str(self.topicPlotCombo.currentText())
            x = self.chatHistoryUser
            y = self.chatHistoryAgent
            interweivedTextList = [a for b in zip(x, y) for a in b]
            joinedChatHistory = ' '.join(interweivedTextList)
            joinedChatHistory=self.cleaner.clean_input(joinedChatHistory)
            if topicClass == 'LDA topics':
                self.topicClassifier.getLDATopicHistogram([joinedChatHistory], self.ax_top)
            elif topicClass == 'M&S topics':
                self.topicClassifier.getMSTopicHistogram([joinedChatHistory], self.ax_top)
            self.PlotFigure.figure.tight_layout()
            self.PlotFigure.canvas.draw()

    def analyzeSentiment(self):
        if self.analyzeSentimentCheck.isChecked():
            self.ax_bottom.cla()
            self.sentimentAnalyzer.plot_sentiment_for_chat(self.ax_bottom)

    def resetButtonPressed(self):
        self.chatText.clear()
        try:
            self.chatBot.resetChat()
        except AttributeError:
            pass
        self.sentimentAnalyzer = Sentiment()

    def enterButtonPressed(self):
        self.inputQueryTextReturn()

    def levelFactorChanged(self):
        new_level_factor = float(self.levelFactorText.text())
        self.chatBot.changeLevelFactor(new_level_factor)

    def saveModelButtonPressed(self):
        self.chatBot.save()

    def versionChanged(self):
        """
        Run when chat version was changed in combo box
        """
        # Instantiate chat bot object
        version = str(self.versionCombo.currentText())
        if version == 'V0':
            self.chatBot = ChattyV0()
        if version == 'V1':
            self.chatBot = ChattyV1(load_from_file = False, folder = folder)
        if version == 'V2':
            self.chatBot = ChattyV2(load_from_file = True, folder = folder)
        if version == 'V3':
            self.chatBot = ChattyV3(load_from_file = True, folder = folder)

    def get_unfiltered_line(self, df, filtered_num):
        return df.convID.values[np.where(df.convIDFiltered == filtered_num)[0][0]]

    def loadTopicTableData(self):
        """
        Populates the topic table with chat data
        """
        filenameTable=os.path.join(folder,'client_agent_summary3.csv')
        filenameTopics=os.path.join(folder,'05_fastTextConv_MStopicId.txt')
        df  = pd.read_csv(filenameTable, index_col = 0,header = 0)
        MSTopicArray = np.genfromtxt(filenameTopics, delimiter=',')

        convID = []
        self.convMessagePositions = []
        convAgentID = []
        convDate = []

        for convid, conv in df.groupby(['convID']):
            convID.append(convid)
            self.convMessagePositions.append(conv['linePos'].values)
            convAgentID.append(conv['agentID'].values[0])
            convDate.append(conv['convDate'].values[0])

        print(len(convDate))
        print(len(convID))
        print(len(convAgentID))

        # Account for filtered out chats
        topicDict = {self.get_unfiltered_line(df,i): topic_modeling.MS_topic_labels[topic]
                     for i, topic in enumerate(MSTopicArray.astype(int).tolist())}

        topics = ['']*len(convID)
        for i, t in enumerate(topics):
            if i in topicDict:
                topics[i] = topicDict[i]

        print(len(topics))

        data = {'convId': convID,'convDate': convDate,'convAgentID': convAgentID,
                'MStopic': topics}

        convDf = pd.DataFrame(data, columns=['convId', 'convDate', 'convAgentID', 'MStopic'])
        #self.datatable = QtGui.QTableWidget(parent=self)
        self.topicTable.setColumnCount(len(convDf.columns))
        self.topicTable.setRowCount(len(convDf.index))
        for i in range(len(convDf.index)):
            for j in range(len(convDf.columns)):
                self.topicTable.setItem(i,j,QtGui.QTableWidgetItem(str(convDf.iat[i, j])))

    def tableCellClicked(self, row, column):
        """
        Table item was clicked.
        Reset chat history, clear text box, load messages from database
        print them in text box and add them to chat history.
        Finally, classify topic if required
        """
        self.chatText.clear()
        self.clearChatHistory()
        self.sentimentAnalyzer = Sentiment()
        print("Row %d and Column %d was clicked" % (row, column))

        messagesIndexes = self.convMessagePositions[row]
        for i in messagesIndexes:
            clientMsg = mdb.get_client_message_from_database(i, self.db)
            agentMsg = mdb.get_agent_message_from_database(i, self.db)
            self.printClientMessage(clientMsg)
            self.printAgentMessage(agentMsg)
            self.updateChatData(clientMsg, agentMsg)

        self.classifyTopic()
        self.analyzeSentiment()

    def printClientMessage(self, msg):
        self.chatText.append('<font color="blue">&lt;client&gt; '+msg+'</font>')
        self.chatText.append('')
        self.chatText.show()

    def printAgentMessage(self, msg):
        self.chatText.append('<font color="green">&lt;agent&gt; '+msg+'</font>')
        self.chatText.append('')
        self.chatText.show()

    def clearChatHistory(self):
        self.chatHistoryUser = []
        self.chatHistoryAgent = []

import sys

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    window = ChatGuiMainWindow()

    app.exec_()

    #sys.exit(app.exec_())
