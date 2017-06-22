#!/usr/bin/env python
# coding: utf-8

input_file_name = '04_doc2vecTrainingDataFiltered'
outpt_file_name = input_file_name + "_tfidfFiltered"


### set paths
import config_project as cfg
data_dir = "/home/walter/Dropbox/S2DS - M&S/Data"
input_file_path = data_dir + "/" + input_file_name + ".txt"

import codecs
chats          = codecs.open(input_file_path,'r','utf-8')

import tfidf_tools
tfidf_tools.show_DF_histogram(chats)
#tfidf_tools.show_TFIDF_histogram(chats)
chats.close()
