#!/usr/bin/env python
# coding: utf-8

input_file_name = '04_doc2vecTrainingDataFiltered'
outpt_file_name = input_file_name + "_TFIDFfiltered"

print('filtering ' + input_file_name)

### set paths
import config_project as cfg
data_dir = "/home/walter/Dropbox/S2DS - M&S/Data"
input_file_path = data_dir + "/" + input_file_name + ".txt"
outpt_file_path = data_dir + "/" + outpt_file_name  + ".txt"

import codecs
chats          = codecs.open(input_file_path,'r','utf-8')
chats_filtered = codecs.open(outpt_file_path,'w','utf-8')

import frequency_tools
topic_related_tokens = frequency_tools.find_topic_related_tokens(chats)
print topic_related_tokens[:10]

# rewind chats file
chats.seek(0)
for line in chats:
    #if i%100 == 0: print('Line ' + str(i))
    new_line = ''
    for word in line.strip().split():
        if word in topic_related_tokens: 
            new_line += ' ' + word
    chats_filtered.write(new_line + '\n')

chats.close()
chats_filtered.close()

print('filtered chats written in ' + outpt_file_path) 
