import nltk
from architecture.utils import *
class HMMSupervisedTagger():
    def __init__(self, x_train, y_train):
        self.xTrain = x_train
        self.yTrain = y_train
        self.trainer = nltk.HiddenMarkovModelTrainer()
        self.tagger = None
    
    def train(self):
        trainer = nltk.HiddenMarkovModelTrainer()
        self.tagger = trainer.train_supervised(tuple_xy4nltk(self.xTrain, self.yTrain))

    def tag(self, x_test):
        return [tag[1] for tag in self.tagger.tag(x_test)]

    def f1_by_tags(self, x_test, y_test):
        tagged_res = list(map(lambda x: self.tag(x), x_test))

        f1_resultant = f1_by_tags(tagged_res, y_test)
        return f1_resultant
        
