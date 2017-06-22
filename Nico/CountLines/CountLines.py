# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 20:36:42 2016

@author: Client
"""



import os, sys
folder = os.path.join('C:/','Users', 'Client', 'Dropbox', 'S2DS - M&S', 'Data')
# Open a file
path = os.path.join(folder,'*.txt' )
dirs = os.listdir( folder )

# This would print all the files and directories
for file in dirs:
    if file.endswith(".txt"):
        with open(os.path.join(folder,file)) as f:
            num_lines = sum(1 for line in f)
            
        print file, num_lines