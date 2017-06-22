import gensim
from gensim.models import Word2Vec
import numpy as np

def _map_vectors(self, toks, model):
        words = []
        for w in toks:
            try:
                # grab the word vector from gensim's innards
                words.append(model.syn0[model.vocab[w].index])
            except KeyError:
                # print "word ignored: {}".format(unicode(w.decode("utf-8")))
                # ignore words that aren't in the model
                continue
        return words

    def _vectorize(self, x, model):
        # split sentence and strip puntuation, newlines etc.
        toks = self.prep_sentence(x)
        # vectorise words, average those vectors to a single vector, then scale to unit length
        a = gensim.matutils.unitvec(np.array(self._map_vectors(toks, model)).mean(axis=0))
        if np.isnan(a).any():
            # skip but remember that there was another sentence here
            return False
        return a, toks

    def average_vectors(self, w2v_model, data, targets):
        # NEED TO EITHER INPUT 0X100 ARRAYS OR REMOVE TARGETS AT SAME INDEX.
        _data = []
        _targets = []
        c = 0
        for idx, i in enumerate(data):
            avg_vector = self._vectorize(' '.join(i), w2v_model)
            if avg_vector is False:
                c+=1
                continue
            else:
                _data.append(avg_vector[0])
                _targets.append(targets[idx])
        print "{} verbatims skipped".format(c)
        print len(_data)
        print len(_targets)
        return _data, _targets