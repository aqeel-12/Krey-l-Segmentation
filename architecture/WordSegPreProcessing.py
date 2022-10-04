
from collections import Counter
import numpy as np

class WordSegPreProcessing():
    def __init__(self, x,y,y_reformat_func=None):
        self.lower = True
        self.y_reformat_func = y_reformat_func

        self.x, self.y = self.extract_training_pairs(x,y)

        self.vocab = list(set(w for i in self.x for w in i))
        self.index_vocab = {w: i for i,
                            w in enumerate(self.vocab)}  # vocab index




    def extract_training_pairs(self,x,y):
        if self.lower:
            x = list(map(lambda z: z.lower(), x))
        y = list(map(lambda z: self.y_reformat_func(z), y))
        return [[*i] for i in x], y
 
        
    def create_n_gram(self,corpus, n=2):
        n_gram = []
        for i in range(len(corpus)-n+1):
            n_gram.append(tuple(corpus[i+j] for j in range(n)))
        return n_gram


        
        




