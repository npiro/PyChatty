# Add app folder to PYTHONPATH

from vectorisations.vectorization import Word2Vec, SentenceVectorizer, SearchTree

import os.path

filename = os.path.join("/Users","piromast","Dropbox","S2DS - M&S","Data","04_clientMessagesFiltered.txt")
W2V = Word2Vec(filename)    # Train word2vec with file in filename
vectorizer = SentenceVectorizer(W2V)    # make sentence vectorizer object
matrix = vectorizer.get_sentences_matrix(filename)  # calculate matrix of sentence vectors of whole text file
tree = SearchTree(matrix)   # Generate search tree (to find nearest vector)

# Lookup example    
v = vectorizer.get_sentence_vector("I lost my order")   # Convert sentence to vector
index, distance = tree.findNearestVector(v)   