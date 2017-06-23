from __future__ import print_function, division
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from nltk.sentiment import vader
from sentiment_parallel import get_sentiment_for_sentence

class Sentiment(object):
    """
    Class for predicting sentiment for client and agent sentences,
    and whole chats. It also provides methods for plotting these.

    Example:
    -------
    %cd ~/ms/Daniel/
    from sentiment import sentiment
    import matplotlib.pyplot as plt
    import seaborn
    %matplotlib qt

    s = sentiment.Sentiment()

    s.add_client_sentence("Hey I need to amend my order becuase I think it is late anyway..")
    s.add_agent_sentence("I'm happy to help you with that, can I have your order number?")
    s.add_client_sentence("That's nice of you, thanks so muuch :) It's EFAS123123")
    s.add_agent_sentence("I get it, what  can I add to it?")
    s.add_client_sentence("Can you add a new skirt I browsed over in your webiste?")
    s.add_agent_sentence("Sure. It's all done.")
    s.add_client_sentence("Fantastic, thanks so much!")

    print ('overall agent sentiment: ', s.get_overall_sentiment_for_agent_sentences())
    print ('overall client sentiment: ', s.get_overall_sentiment_for_client_sentences())
    print ('sentiments of agent sentences: ', s.get_sentiments_for_client_sentences())
    print ('sentiments of client sentences: ', s.get_sentiments_for_client_sentences())
    s.plot_sentiment_for_chat()
    """

    def __init__(self):
        self.client_msgs = []
        self.agent_msgs = []
        self.client_msgs_scores = []
        self.agent_msgs_scores = []
        self.sentiment_analyzer = vader.SentimentIntensityAnalyzer()

    def add_client_sentence(self, text):
        """
        Adds a new client sentence to the conversation.
        :param text: [string], input string
        """
        self.client_msgs.append(text)

    def add_agent_sentence(self, text):
        """
        Adds a new agent sentence to the conversation.
        :param text: [string], input string
        """
        self.agent_msgs.append(text)

    def add_sentence(self, text, actor='client'):
        """
        Adds a new agent sentence to either client or agent.
        :param text: [string], input string
        :param actor: [string], client or agent.
        """
        if actor == 'client':
            self.add_client_sentence(text)
        else:
            self.add_agent_sentence(text)

    def get_sentiments_for_list_of_sentences(self, list_of_text):
        """
        Returns the sentiment for a list of sentences. Executes in
        parallel using joblib.
        :param list_of_text: [list], list of input strings
        :return: array of compound sentiment score: -1 is neg, 0 is
                 neutral, 1 is pos
        """
        sentiments = np.array(Parallel(n_jobs=-1)
                              (delayed(get_sentiment_for_sentence)
                              (text, self.sentiment_analyzer)
                               for text in list_of_text))
        return sentiments

    def get_sentiments_for_client_sentences(self):
        """
        Returns the sentiment for the client sentences. Executes in
        parallel using joblib, only predicts sentences that have not
        been predicted.
        :return: array of compound sentiment score for client sentences.
                 -1 is neg, 0 is neutral, 1 is pos
        """
        if len(self.client_msgs) == 0:
            raise ValueError('List of client messages is empty.')

        # we have sentences to predict
        if len(self.client_msgs) > len(self.client_msgs_scores):
            list_of_text = self.client_msgs[len(self.client_msgs_scores):]
            sentiments = np.array(Parallel(n_jobs=-1)
                                  (delayed(get_sentiment_for_sentence)
                                  (text, self.sentiment_analyzer)
                                   for text in list_of_text))
            self.client_msgs_scores.extend(list(sentiments))

        # all sentences are predicted
        return self.client_msgs_scores

    def get_sentiments_for_agent_sentences(self):
        """
        Returns the sentiment for the agent sentences. Executes in
        parallel using joblib, only predicts sentences that have not
        been predicted.
        :return: array of compound sentiment score for client sentences.
                 -1 is neg, 0 is neutral, 1 is pos
        """
        if len(self.agent_msgs) == 0:
            raise ValueError('List of client messages is empty.')

        # we have sentences to predict
        if len(self.agent_msgs) > len(self.agent_msgs_scores):
            list_of_text = self.agent_msgs[len(self.agent_msgs_scores):]
            sentiments = np.array(Parallel(n_jobs=-1)
                                  (delayed(get_sentiment_for_sentence)
                                  (text, self.sentiment_analyzer)
                                   for text in list_of_text))
            self.agent_msgs_scores.extend(list(sentiments))

        # all sentences are predicted
        return self.agent_msgs_scores

    def get_overall_sentiment_for_list_of_sentences(self, list_of_text):
        """
        Returns overall sentiment for a list of sentences.
        :param list_of_text: [list], list of input strings
        :return: overall compound sentiment score of the list_of_text:
                -1 is neg, 0 is neutral, 1 is pos
        """
        if len(list_of_text) > 0:
            text = ' '.join(list_of_text)
            sentiment = get_sentiment_for_sentence(text, self.sentiment_analyzer)
            return sentiment
        else:
            raise ValueError('List of sentences has length 0.')

    def get_overall_sentiment_for_client_sentences(self):
        """
        Returns overall sentiment for client sentences.
        :return: overall compound sentiment score of client sentences:
                -1 is neg, 0 is neutral, 1 is pos
        """
        if len(self.client_msgs) > 0:
            text = ' '.join(self.client_msgs)
            sentiment = get_sentiment_for_sentence(text, self.sentiment_analyzer)
            return sentiment
        else:
            raise ValueError('List of client messages is empty.')

    def get_overall_sentiment_for_agent_sentences(self):
        """
        Returns overall sentiment for agent sentences.
        :return: overall compound sentiment score of agent sentences:
                -1 is neg, 0 is neutral, 1 is pos
        """
        if len(self.agent_msgs) > 0:
            text = ' '.join(self.agent_msgs)
            sentiment = get_sentiment_for_sentence(text, self.sentiment_analyzer)
            return sentiment
        else:
            raise ValueError('List of agent messages is empty.')

    def get_overall_sentiment_for_chat(self):
        """
        Returns overall sentiment for the whole conversation, i.e.
        client and agent sentences together.
        :return: overall compound sentiment score of client and
                 agent sentences: -1 is neg, 0 is neutral, 1 is pos
        """
        if len(self.client_msgs) > 0 and len(self.agent_msgs) > 0:
            text = ' '.join(self.client_msgs)
            text += ' ' + ' '.join(self.agent_msgs)
            sentiment = get_sentiment_for_sentence(text, self.sentiment_analyzer)
            return sentiment
        else:
            raise ValueError('List of client or agent messages is empty.')

    def plot_sentiment_for_chat(self):
        """
        Plots the sentiment of the conversation (both client and agent) in
        a single plot.
        :return ax: [obj], a matplotlib axes object we can plot to.
        """
        if len(self.client_msgs) > 0 and len(self.agent_msgs) > 0:
            # update sentiment scores
            _ = self.get_sentiments_for_client_sentences()
            _ = self.get_sentiments_for_agent_sentences()

            # if the length of the two sentence lists differ make them equal
            client_len = len(self.client_msgs_scores)
            agent_len = len(self.agent_msgs_scores)
            client_msgs = self.client_msgs_scores
            agent_msgs = self.agent_msgs_scores

            if client_len > agent_len:
                diff = client_len - agent_len
                agent_msgs += [np.nan] * diff
            elif client_len < agent_len:
                diff = agent_len - client_len
                client_msgs += [np.nan] * diff

            df = pd.DataFrame(zip(client_msgs, agent_msgs), columns=['Client', 'Agent'])
            ax = df.plot()
            return ax
        else:
            raise ValueError('List of client or agent messages is empty.')