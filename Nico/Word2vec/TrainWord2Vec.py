from gensim.models import Word2Vec
 
def w2v(self, save_to=False):
    self.report("Training w2v model")
    if len(self.data) == 0:
        raise RuntimeError("No data has been loaded, load data using .csv method")
    model = Word2Vec(self.data, size=200, window=7, min_count=5, workers=12, iter=6)  # Window, context window
    if save_to:
        model.save(save_to)
    return model

def load_w2v(self, load_from):
    model = Word2Vec.load(load_from)
    return model