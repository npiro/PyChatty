#!/usr/bin/env python
# coding: utf-8

"""
Attemp to correct typos from input_file
"""

import sys
if len(sys.argv) != 3: 
    raise ValueError('Usage: python correct_typos.py data_dir input_file_name')
data_dir        = sys.argv[1]
input_file_name = sys.argv[2]
print('Correcting typos from ' + input_file_name)



outpt_file_name = input_file_name + "_TyposCorrected"
input_file_path = data_dir + "/" + input_file_name + ".txt"
outpt_file_path = data_dir + "/" + outpt_file_name  + ".txt"


import codecs
chats          = codecs.open(input_file_path,'r','utf-8')
chats_filtered = codecs.open(outpt_file_path,'w','utf-8')


from spelling_tools import *


processing_order = 'serial'
#processing_order = 'parallel'

if processing_order == 'serial':
    i=0
    rescued_typos = 0
    for line in chats:
        i += 1
        if i%1 == 0: print('Line ' + str(i))
        new_line = ''
        for word in line.strip().split():
            
            corrected_word, rescued_typo = my_spell(word)
            rescued_typos += rescued_typo
            new_line      += ' ' + corrected_word

        chats_filtered.write(new_line + '\n')
        
    print('rescued_typos = ' + str(rescued_typos))

elif processing_order == 'parallel':

    from joblib import Parallel, delayed
    import multiprocessing
    num_cores = multiprocessing.cpu_count()
    print 'num_cores = ',num_cores

    i=0
    for line in chats:
        i += 1
        if i%10 == 1: print('Line ' + str(i))
        words = line.strip().split()
    
        new_words = Parallel(n_jobs=num_cores-1)\
        (delayed(my_spell)(word) for word in words)

        new_line = ' '.join(new_words)
        #print new_line 
        chats_filtered.write(new_line + '\n')


chats.close()
chats_filtered.close()

print('filtered chats written in ' + outpt_file_path) 
