# -*- coding: utf-8 -*-
"""
Set of functions to create and retrieve messages from data frame or data base.
create_data_frame(agent_filename, client_filename): creates a message dataframe
save_data_frame(df, filename = 'MessagesDF.pkl', folder = '.'): saves dataframe to file
load_data_frame(df, filename, folder = '.'): loads dataframe from file
create_database(agent_filename, client_filename, db_filename, folder): create message database
get_client_message_from_dataframe(df, index): get client message at index from dataframe
get_agent_message_from_dataframe(df, index): get agent message at index from dataframe
get_client_message_from_database(index, db_engine): get client message from database
get_agent_message_from_database(index, db_engine): get agent message from database
connect_to_db(db_filename): connect to database
disconnect_from_db(db_engine): disconnect from database
"""

import pandas as pd
import codecs
import os.path
import pickle
import sqlite3

def create_data_frame(agent_filename, client_filename):
    """
    Creates a pandas data frame with two columns: "Agent" and "Client"
    containing agent and cliente messages.
    Each row corresponds to one question/answer pair
    return the pandas data frame
    """
    with codecs.open(agent_filename, 'r', 'utf-8') as fA, codecs.open(client_filename, 'r', 'utf-8') as fC:
        df = pd.DataFrame({'Agent':[lines for lines in fA], 'Client':[lines for lines in fC]})
    
    return df

def save_data_frame(df, filename = 'MessagesDF.pkl', folder = '.'):
    """
    Saves messages data frame using pickle
    Arguments:
    df: pandas data frame containing messages
    filename: name of file to save (default = 'MessagesDF.pkl')
    folder: folder to save file to (default = '.')
    """
    
    fn = os.path.join(folder,filename)
    with open(fn, 'wb') as fid:
        pickle.dump(df, fid, protocol=2)

def load_data_frame(filename, folder = '.'):
    """
    Load messages data frame from file
    """
    fn = os.path.join(folder,filename)
    with open(fn, 'rb') as fid:
        df = pickle.load(fid)
        
    return df
        
    

def create_database(agent_filename, client_filename, db_filename, folder):
    """
    Creates an SQL database containing messages.
    Database will contain two columns: 'Agent' and 'Client'
    with agent and client messages respectively, in quesiton/answer pairs
    """
    agent_filename = os.path.join(folder, agent_filename)
    client_filename = os.path.join(folder, client_filename)
    db_filename = os.path.join(folder, db_filename)
    
    df = create_data_frame(agent_filename, client_filename)
        

    # create DB engine
    engine = sqlite3.connect(db_filename)

    # Write dataframe to database
    df.to_sql("messages", engine, chunksize = 4096)
    
def get_client_message_from_dataframe(df, index):
    """
    Get client message from data frame:
    df: Pandas data frame object
    index: row index (integer) of message
    """
    return df.Client.iloc[index].strip()
    
def get_agent_message_from_dataframe(df, index):
    """
    Get agent messages from data frame:
    df: Pandas data frame object
    index: row index (integer) of message
    """
    return df.Agent.iloc[index].strip()
    
def get_client_message_from_database(index, db_engine):
    """
    Get the client message at index from data base
    index: Integer specifying message row in db
    db_engine: data base engine object (use connect_to_db to obtain object)
    """
    df = pd.read_sql_query('SELECT Client FROM messages LIMIT '+str(index)+',1', db_engine)
    return df.Client.iloc[0].strip()
      
def get_agent_message_from_database(index, db_engine):
    """
    Get the client message at index from data base
    index: Integer specifying message row in db
    db_engine: data base engine object (use connect_to_db to obtain object)
    """
    df = pd.read_sql_query('SELECT Agent FROM messages LIMIT '+str(index)+',1', db_engine)
    return df.Agent.iloc[0].strip()
    
def connect_to_db(db_filename):
    """
    connect to database in db_filename.
    Returns database engine object
    """
    engine = sqlite3.connect(db_filename)
    
    return engine
    
def disconnect_from_db(db_engine):
    """
    disconnect from data base
    db_engine: database engine object
    """
    db_engine.close()
    
    
    
if __name__ == "__main__":
    folder = os.path.join('c:/','Users','Client','Dropbox','S2DS - M&S','Data')
    AgentMessagesFile = os.path.join(folder, '03_agentMessages.txt')
    ClientMessagesFile = os.path.join(folder, '03_clientMessages.txt')
    MessageDataBase = os.path.join(folder, 'MessagesDB.db')
    
    # Create messages data frame
    messages_dataframe = create_data_frame(AgentMessagesFile, ClientMessagesFile)
    
    # Save data frame
    save_data_frame(messages_dataframe, filename = 'MessagesDF.pkl', folder = folder)
    
    # Create data base
    create_database(AgentMessagesFile, ClientMessagesFile, MessageDataBase, folder)