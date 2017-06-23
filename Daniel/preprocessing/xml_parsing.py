"""
This submodule holds methods for extracting client, agent messages and full
conversations from the XML file we got from M&S. Ultimately it creates training
data for both Word2Vec and Doc2Vec.
"""

import codecs
import os
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
import pandas as pd


def get_client_agent_conversations(data_folder, input_file, client_file, agent_file, conv_file):
    """
    Goes through a cleaned XML and generates four .txt files. One for client
    messages, one for agent messages, one which holds one conversation per
    line and a copy of the last one, where all sentences are separated by ||.
    There's a one-to-one correspondence between the client and agent files
    so the e.g. the 3rd line of the agent file is the answer the customer
    service agent gave to the 3rd line of the client  file.

    If the agent or client multiple messages consequtively they will be merged
    so they appear on the same line.

    The client and agent files are for querying the model and finding the
    closest question-answer pair in the corpus.

    The conversation file is for training the Doc2Vec model.

    It will also create a file with the name client_agent_summary.csv which is
    a pandas DataFrame saved as a csv file holding the line numbers (will used
    to keep a correspondence between filtered files and original ones), ID of
    conversation, length of conversation, and the position of each line within
    the conversation. Also the date, conversation length in secounds and the
    agent ID will be saved to this file.

    This script also ensures that all client-agent msg pairs are unique in the
    whole corpus.

    :param data_folder: [string], absolute path to data folder
    :param input_file:  [string], name of cleaned XML file
    :param client_file: [string], name of client file output
    :param agent_file: [string], name of agent file output
    :param conv_file: [string], name of conversation file output
    :return: None
    """

    # open files for reading and writing
    input_file = codecs.open(os.path.join(data_folder, input_file), 'rU', 'utf-8')
    client_msgs_file = codecs.open(os.path.join(data_folder, client_file), 'w', 'utf-8')
    agent_msgs_file = codecs.open(os.path.join(data_folder, agent_file), 'w', 'utf-8')
    summary_file = os.path.join(data_folder, 'client_agent_summary.csv')

    conv_msgs_file = codecs.open(os.path.join(data_folder, conv_file), 'w', 'utf-8')
    filename, extension = os.path.splitext(conv_file)
    conv_msgs_file2 = filename + '2' + extension
    conv_msgs_file2 = codecs.open(os.path.join(data_folder, conv_msgs_file2), 'w', 'utf-8')

    client_agent_cache = {}
    conv_id = []
    conv_length = []
    conv_dates = []
    conv_secs = []
    agent_personal_ids = []
    conv_position = []
    conv_count = 0

    for i, line in enumerate(input_file):
        # ignore xml declaration and root line
        if i > 1:
            # define each line as a new xml object
            chat = BeautifulSoup(line, 'xml')

            # define main variables
            parties = chat.findAll('newParty')
            if len(parties) > 0:
                message_pairs = []
                conv_date = chat.chatTranscript['startAt']
                client_id = ''
                agent_id = ''
                agent_personal_id = ''
                external_id = ''
            
            # work out which message belongs to whom: client, agent
            for party in parties:
                if party.userInfo['userType'] == 'CLIENT':
                    client_id = party['userId']
                elif party.userInfo['userType'] == 'AGENT':
                    agent_id = party['userId']
                    agent_personal_id = party.userInfo['personId']
                elif party.userInfo['userType'] == 'EXTERNAL':
                    external_id = party['userId']
                    if agent_personal_id == '':
                        agent_personal_id = party.userInfo['personId']
                    else:
                        agent_personal_id += '|' + party.userInfo['personId']

            # get all messages from the chat and find length of chat in secs
            messages = chat.findAll('message')

            # define flags for merging consecutive messages from the same party
            first_client_msg = 0
            client_msg = 0
            agent_msg = 0

            # do we have both client and agent/external agent?
            if client_id != '' and (agent_id != '' or external_id != '') and len(messages) > 0:
                conv_sec = messages[-1]['timeShift']
                for message in messages:
                    msg_text = message.msgText.contents[0]
                    msg_id = message['userId']

                    # we look for the first client message and start with that
                    if msg_id == client_id and first_client_msg == 0:
                        first_client_msg = 1
                        client_msg = 1
                        client_msg_text = msg_text

                    # CLIENT MESSAGE
                    if msg_id == client_id and first_client_msg == 1:
                        # agent has spoken and now it's client again, save tuple
                        if agent_msg == 1:
                            client_agent_msg = (client_msg_text, agent_msg_text)
                            if client_agent_msg not in client_agent_cache:
                                client_agent_cache[client_agent_msg] = 1
                                message_pairs.append(client_agent_msg)
                            agent_msg = 0

                        # if no previous client msg define new client_msg text
                        if client_msg == 0:
                            client_msg_text = msg_text
                        # else continue previous
                        elif client_msg == 1 and msg_text != client_msg_text:
                            client_msg_text += ' ' + msg_text
                        client_msg = 1

                    # AGENT MASSAGE
                    if msg_id in [agent_id, external_id] and first_client_msg == 1:
                        client_msg = 0
                        # if no previous agent msg define new client_msg text
                        if agent_msg == 0:
                            agent_msg_text = msg_text
                        # else continue previous
                        else:
                            agent_msg_text += ' ' + msg_text
                        agent_msg = 1

                # if the very last tuple wasn't saved save it
                if agent_msg == 1:
                    client_agent_msg = (client_msg_text, agent_msg_text)
                    if client_agent_msg not in client_agent_cache:
                        client_agent_cache[client_agent_msg] = 1
                        message_pairs.append(client_agent_msg)

                # write out sentences from conversation
                if len(message_pairs) != 0:
                    for msg in message_pairs:
                        client_msgs_file.write(msg[0] + '\n')
                        agent_msgs_file.write(msg[1] + '\n')
                        conv = msg[0] + ' ' + msg[1]
                        conv_msgs_file.write(conv + ' ')
                        conv_msgs_file2.write(' || '.join(sent_tokenize(conv)))

                    # one conversation per line in doc2vec train data
                    conv_msgs_file.write('\n')
                    conv_msgs_file2.write('\n')

                    # save variables for summary
                    conv_len = len(message_pairs)
                    conv_length.extend([conv_len] * conv_len)
                    conv_id.extend([conv_count] * conv_len)
                    conv_position.extend(range(conv_len))
                    conv_dates.extend([conv_date] * conv_len)
                    conv_secs.extend([conv_sec] * conv_len)
                    agent_personal_ids.extend([agent_personal_id] * conv_len)
                    conv_count += 1

    # prepare DataFrame for summary and save it
    data = zip(conv_id, conv_position, conv_length, conv_dates, conv_secs, agent_personal_ids, range(len(conv_length)))
    df = pd.DataFrame(data, columns=['convID', 'convPos', 'convLen', 'convDate', 'convSec', 'agentID', 'linePos'])
    df.to_csv(summary_file)

    # close files
    input_file.close()
    client_msgs_file.close()
    agent_msgs_file.close()
    conv_msgs_file.close()
    conv_msgs_file2.close()


def get_w2v_training(data_folder, input_file, output_file):
    """
    Goes through a cleaned XML and saves all messages (can be multiple
    consecutive lines of text from client or agent) in a single output file for
    training the Word2Vec model later.

    Each line is a single message from either a client or agent. Unlike with
    get_client_agent_conversations, here we include conversations with  a single
    party as well to maximize the size of our training data.

    It ensures that all sentences are unique in the corpus.

    :param data_folder: [string], absolute path to data folder
    :param input_file:  [string], name of cleaned XML file
    :param output_file: [string], name of the output file
    :return: None
    """

    # open files for reading and writing
    input_file = codecs.open(os.path.join(data_folder, input_file), 'rU', 'utf-8')
    output_file = codecs.open(os.path.join(data_folder, output_file), 'w', 'utf-8')
    msg_cache = {}

    for i, line in enumerate(input_file):
        # ignore xml declaration and root line
        if i > 1:
            # define each line as a new xml object
            chat = BeautifulSoup(line, 'xml')
            first_msg = 0

            # get all messages from the chat
            messages = chat.findAll('message')
            msg_out = ''
            msg_id_old = ''

            for message in messages:
                # variables for tracking the msg through conversation
                msg_text = message.msgText.contents[0]
                msg_id = message['userId']

                if msg_id != msg_id_old:
                    # new msg
                    msg_id_old = msg_id

                    # very first msg?
                    if first_msg == 0:
                        first_msg = 1
                        msg_out = msg_text
                    else:
                        # end of prev msg, if not in cache write it out
                        if msg_out not in msg_cache:
                            msg_cache[msg_out] = 1
                            output_file.write(msg_out)
                        # start new msg
                        msg_out = '\n' + msg_text
                else:
                    # continue previous message
                    msg_out += ' ' + msg_text

    # close files
    input_file.close()
    output_file.close()
