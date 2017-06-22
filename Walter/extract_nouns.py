#!/usr/bin/env python
# coding: utf-8
"""
Create corpus nouns file  
"""
import os, nltk, codecs
data_folder = '/home/walter/Dropbox/S2DS - M&S/Data/'
input_file = '03_clientMessages.txt'
output_file = '03_clientMessagesOnlyNouns.txt'
input_file = codecs.open(os.path.join(data_folder, input_file), 'rU', 'utf-8')
output_file = codecs.open(os.path.join(data_folder, output_file), 'w', 'utf-8')

for i, line in enumerate(input_file):
    if i%100 == 0: print('Line ' + str(i))
    new_line = ''
    for word_type in nltk.pos_tag(nltk.word_tokenize(line)):
        if word_type[1] in ['NN', 'NNP', 'NNPS', 'NNS']:
            word = word_type[0].lower()
            if len(word)>1:
                new_line += ' ' + word
    output_file.write(new_line + '\n')

input_file.close()
output_file.close()
