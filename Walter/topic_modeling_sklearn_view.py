
# coding: utf-8

# ### Load existing LDA model

# In[1]:

import topic_modeling
topic_modeling.verbose  = 'yes'
model = topic_modeling.load_LDA_model('/home/walter/Dropbox/S2DS - M&S/Data/LDAmodel__num_topics30__passes5__min_df5__max_df0.95__max_featuresNone.pickle')




# ### Calculate and plot topic distributions for a given chat

# In[20]:

chat = ["sparks code offer"]

lda_topic_distribution = topic_modeling.get_lda_topic_distribution(model,chat)
MS_topic_distribution  = topic_modeling.get_MS_topic_distribution(model,chat)
print MS_topic_distribution
print topic_modeling.get_MS_topic_labels()

#from matplotlib.pylab import plt
#get_ipython().magic(u'matplotlib inline')
#plt.figure(num=None, figsize=(12, 4))

#plt.subplot(1,2,1)
#plt.hist(lda_topic_distribution,range(len(lda_topic_distribution)))
#plt.xlabel('LDA Topic')
#plt.ylabel('Topic probability')

#plt.subplot(1,2,2)
#plt.hist(MS_topic_distribution,range(len(MS_topic_distribution)))
#plt.xlabel('M&S Topic')
#plt.ylabel('Topic probability')


## ### See top words for topic

## In[14]:

#n_top_words = 20
#def print_top_words(model, feature_names, n_top_words):
    #for topic_idx, topic in enumerate(model.components_):
        #print("Topic #%d:" % topic_idx)
        #print(" ".join([feature_names[i]
                        #for i in topic.argsort()[:-n_top_words - 1:-1]]))
    #print()
#print("\nTopics in LDA model:")
#tf_feature_names = model['tf_vectorizer'].get_feature_names()
#print_top_words(model['lda_tf'], tf_feature_names, n_top_words)

