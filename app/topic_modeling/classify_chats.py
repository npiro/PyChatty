#!/usr/bin/env python
# coding: utf-8
"""
"""

import sys
if len(sys.argv) != 3: 
    raise ValueError('Usage: python classify_chats.py data_dir input_file_name')
data_dir        = sys.argv[1]
input_file_name = sys.argv[2]
print('Correcting typos from ' + input_file_name)



outpt_file_name = input_file_name + "_MStopicId"
input_file_path = data_dir + "/" + input_file_name + ".txt"
outpt_file_path = data_dir + "/" + outpt_file_name + ".txt"

print input_file_path



import codecs
chats          = codecs.open(input_file_path,'r','utf-8')
chats_filtered = codecs.open(outpt_file_path,'w','utf-8')

import topic_modeling
classy = topic_modeling.LDATopicClassifier(folder='/home/walter/Dropbox/S2DS - M&S/Data')

processing_order = 'serial'
#processing_order = 'parallel'

if processing_order == 'serial':
    i=0
    rescued_typos = 0
    for line in chats:
        i += 1
        if i%10 == 0: print('Line ' + str(i))
        MStopic_id = classy.get_predominant_MStopic_id([line])
        chats_filtered.write(str(MStopic_id) + '\n')
        
        #print line
        #print MStopic_id
        #print classy.get_predominant_MStopic_label(line)
        #print classy.getMSTopicDistri([line])

#elif processing_order == 'parallel':

    #from joblib import Parallel, delayed
    #import multiprocessing
    #num_cores = multiprocessing.cpu_count()
    #print 'num_cores = ',num_cores

    #i=0
    #for line in chats:
        #i += 1
        #if i%10 == 1: print('Line ' + str(i))
        #words = line.strip().split()
    
        #new_words = Parallel(n_jobs=num_cores-1)\
        #(delayed(my_spell)(word) for word in words)

        #new_line = ' '.join(new_words)
        ##print new_line 
        #chats_filtered.write(new_line + '\n')


chats.close()
chats_filtered.close()

print('filtered chats written in ' + outpt_file_path) 
