
from collections import Counter
import numpy as np


class WordSegPreProcessing():
    def __init__(self, x,y,y_reformat_func=None):
        self.lower = True
        self.y_reformat_func = y_reformat_func

        self.x, self.y = self.extract_pairs(x,y)

        self.vocab = list(set(w for i in self.x for w in i))
        self.all_tags = list(set(w for i in self.y for w in i))
        self.index_vocab = {w: i for i,
                            w in enumerate(self.vocab)}  # vocab index
        self.index_tag = {t: i for i,t in enumerate(self.all_tags)}
        self.index2tag = {i: t for i, t in enumerate(self.all_tags)}



    def extract_pairs(self,x,y):
        if self.lower:
            x = list(map(lambda z: z.lower(), x))
        y = list(map(lambda z: self.y_reformat_func(z), y))
        return [[*i] for i in x], y
    
    def apply_features(self, x):
        #The alphabet of Kreyo`l indicates that only certain letters can make a probable
        # grapheme with others. and some can't. Some can't even stand alone.
        # ambiguous = a,n,h,e,g,o,i
        # standalone = the rest
        # cannotstandalone = c,u
        #This list of feature can apply to both train and test

        features_of_x = []
        for i,let in enumerate(x):
            features = {"let": let,
                        "ambi": 0,
                        "standalone": 0,
                        "cannot_standalone": 0,
                        "prev_let": x[i-1]}
            if i == 0:
                features["prev_let"] = None
            if let in ("a","n","e","g","h","o","i"):
                features["ambi"] = 1
            elif let in ("c","u"):
                features["cannot_standalone"] = 1
            else:
                features["standalone"] = 1
            features_of_x.append(frozenset(zip(features.keys(), features.values())))
        return features_of_x
    def generate_features(self, x):
        return list(map(lambda y: self.apply_features(y), x))
    def let2index(self, x):
        return [self.index_vocab[w] for w in x]
    def tag2index(self, y):
        return [self.index_tag[w] for w in y]


"""
    def create_n_gram(self,corpus, n=2):
        n_gram = []
        for i in range(len(corpus)-n+1):
            n_gram.append(tuple(corpus[i+j] for j in range(n)))
        return n_gram
"""

        
        




