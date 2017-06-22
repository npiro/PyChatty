# coding: utf-8

"""
TF-IDF related functions
"""

verbose         = 'yes'
#verbose         = 'no'

import logging
PROGRESS_NUM = 25
logging.addLevelName(PROGRESS_NUM, 'PROGRESS')


def find_topic_related_tokens(chats,min_df=40):
    """
    Filter out tokens with low DF (rare words) and low TF-IDF (commom words) values

    Notice that M&S customer support chats, the rare words are contaminated by different
    sources, e.g. typos, proper names not easily recognizable like foreign names
    
    parameters:
    chats
    min_df : minimum DF value considered 
    """

    #====================================================================================#
    # Configure messages sent to the terminal 
    if verbose=='yes': level=logging.INFO 
    else:              level=PROGRESS_NUM
    logging.basicConfig(format='%(levelname)s : %(message)s', level=level)
    #====================================================================================#
    

    #====================================================================================#
    logging.log(PROGRESS_NUM,'create TF-IDF matrix')
    logging.log(logging.INFO,'ignoring tokens present in less than ' + str(min_df) + ' documents')

    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(min_df=min_df) 
    
    tfidf_matrix = tfidf_vectorizer.fit_transform(chats)
    tfidf_means  = tfidf_matrix.mean(axis=0).transpose()
    #====================================================================================#


    #====================================================================================#
    logging.log(PROGRESS_NUM,'Filter common tokens (with TF-IDF values below the median)')

    import statistics
    tfidf_median =statistics.median(tfidf_means).mean()
    logging.info('TFIDF median = ' + str(tfidf_median))

    threshold_lower = tfidf_median * 30.0
#    threshold_lower = tfidf_median
#    threshold_lower = 0.00001

    indices  = tfidf_vectorizer.idf_.argsort()[::-1] # Sort by TF-IDF.
    features = tfidf_vectorizer.get_feature_names() # Word list.

    topic_related_tokens = list()
    for i in indices:
        if tfidf_means[i] > threshold_lower:
            topic_related_tokens.append(features[i])

    if verbose=='yes':
        print('topic_related_tokens[:10]')
        print(topic_related_tokens[:10])
    #====================================================================================#

    return topic_related_tokens

def get_DF_histogram(chats):

    from collections import Counter

    #calculate words frequencies per document
    word_frequencies = [Counter(text.strip().split()) for text in chats]

    #calculate document frequency
    document_frequencies = Counter()
    map(document_frequencies.update, (word_frequency.keys() for word_frequency in word_frequencies))

#    print(document_frequencies)
    return document_frequencies

    
def get_TFIDF_vector(chats,min_df=4):
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(min_df=min_df)
    X = vectorizer.fit_transform(chats)

    #from matplotlib.pylab import plt
    #plt.hist(vectorizer.idf_)
    #plt.show()
    
    return vectorizer.idf_


def show_top_TFIDF_values(chats):
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(min_df=0.1, max_df=0.9)
    X = vectorizer.fit_transform(chats)

    indices = vectorizer.idf_.argsort()[::-1] # Sort by TF-IDF.
    features = vectorizer.get_feature_names() # Word list.

    # Top 10 words in TF-IDF: (word, TF-IDF value, index).
    top_n = 100
    top_features = [(features[i], vectorizer.idf_[i], i) for i in indices[:top_n]]
    print('top_features'); print(top_features)
