from nltk.util import everygrams, bigrams, trigrams, skipgrams
import nltk
from nltk import NgramTagger
from architecture.utils import f1_by_tags, tuple_xy4nltk

_ngramArg_to_ngramMethod = {
    1: nltk.tag.UnigramTagger,
    2: nltk.tag.BigramTagger,
    3: nltk.tag.TrigramTagger
}


class NGramSupervisedTagger:
    #Supports backoff
    def __init__(self, trainX, trainY, ngram_choice=2):
        self.trainX = trainX
        self.trainY = trainY
        self.tagger = None
        self.ngram_choice = ngram_choice

    def create_n_gram_tagger(self):

        train = tuple_xy4nltk(self.trainX, self.trainY)
        for i in range(1, self.ngram_choice + 1):
            self.tagger = _ngramArg_to_ngramMethod[i](
                train, backoff=self.tagger)

    def tag(self, x_test):
        return [tag[1] for tag in self.tagger.tag(x_test)]

    def f1_by_tags(self, x_test, y_test):
        tagged_res = list(map(lambda x: self.tag(x), x_test))
        
        f1_resultant = f1_by_tags(tagged_res, y_test)
        return f1_resultant

    
