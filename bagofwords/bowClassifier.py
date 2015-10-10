from constants import SENTIMENT
import bowModel

class BowClassifier:
    def __init__(self):
        self.pos_model = bowModel.BowModel()
        self.neg_model = bowModel.BowModel()
        
    def populate(self,pos_file,neg_file):
        self.pos_model = loadWordsFromFile(self, pos_file)
        self.neg_model = loadWordsFromFile(self, neg_file)
        
    def classify(self, text):
        pos_score = pos_model.evaluate(text)
        neg_score = neg_model.evaluate(text)

        if pos_score > neg_score:
            return SENTIMENT.POSITIVE
        elif neg_score > pos_score:
            return SENTIMENT.NEGATIVE

        return SENTIMENT.NEUTRAL
        
