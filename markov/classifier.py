import pickle

from .model import MarkovModel
from constants import SENTIMENT

class MarkovClassifier:
    def __init__(self, order, smoothing):
        self.k = order
        self.smoothing = smoothing
        self.pos_model = MarkovModel(self.k, self.smoothing)
        self.neg_model = MarkovModel(self.k, self.smoothing)

    def trainOnCorpora(self, posfile, negfile):
        self.pos_model.trainOnCorpus(posfile)
        self.neg_model.trainOnCorpus(negfile)
        return 0

    def classify(self, text):
        pos_likelihood = self.pos_model.getProb(text)
        neg_likelihood = self.neg_model.getProb(text)

        if pos_likelihood > neg_likelihood:
            return SENTIMENT.POSITIVE
        elif neg_likelihood > pos_likelihood:
            return SENTIMENT.NEGATIVE
        return SENTIMENT.NEUTRAL


    # Save and Load from File method:
    @staticmethod
    def loadFromBuffer(buffer):
        ##  the pos_model and neg_model will probably not be loaded as they should in this implementation
        ##  resolve pointers?
        ##  Update: actually, it seems like it works. Keep this comment if problem in future though
        return pickle.loads(buffer)

    @staticmethod
    def loadFromFile(filepath):
        with open(filepath, 'rb') as f:
            loadbuf = f.read()
            print("loading %d bytes from file \"%s\"..." % (len(loadbuf), filepath))
            mc =  MarkovClassifier.loadFromBuffer(loadbuf)
            print("done.")
            return mc

    def saveToBuffer(self):
        ##  the pos_model and neg_model will probably not be saved as they should in this implementation
        ##  resolve pointers?
        ##  Update: actually, it seems like it works. Keep this comment if problem in future though
        savebuf = pickle.dumps(self)
        return savebuf

    def saveToFile(self, filepath):
        with open(filepath, 'wb') as f:
            savebuf = self.saveToBuffer()
            print("writing %d bytes to file \"%s\"..." % (len(savebuf), filepath))
            f.write(savebuf)
            print("done.")
