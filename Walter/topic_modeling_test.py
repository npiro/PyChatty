#!/usr/bin/env python
# coding: utf-8

file_name    = '04_doc2vecTrainingDataFiltered'

#tfidf_threshold = 0.3
#num_topics      = 100
print('modeling ' + file_name)

### set paths
import config_project as cfg
input_file_path = cfg.data_dir + "/" + file_name + ".txt"
outpt_file_path = cfg.data_dir + "/" + file_name + "_tfidfFiltered.txt"


import topic_modeling
topic_modeling.verbose         = 'yes'
model = topic_modeling.fit_LDA(input_file_path,\
num_topics=21)
#model = topic_modeling.fit_LDA(input_file_path,\
#num_topics=100,update_every=1,passes=1,chunksize=5000)
