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
        pos_likelihood, posTotalRowMisses, posTotalColMisses = self.pos_model.getProb(text)
        neg_likelihood, negTotalRowMisses, negTotalColMisses = self.neg_model.getProb(text)

        #print("P(pos) = %.4e" % pos_likelihood)
        #print("P(neg) = %.4e" % neg_likelihood)

        if pos_likelihood > neg_likelihood:
            return SENTIMENT.POSITIVE, posTotalRowMisses, posTotalColMisses, negTotalRowMisses, negTotalColMisses
        elif neg_likelihood > pos_likelihood:
            return SENTIMENT.NEGATIVE, posTotalRowMisses, posTotalColMisses, negTotalRowMisses, negTotalColMisses
        return SENTIMENT.NEUTRAL, posTotalRowMisses, posTotalColMisses, negTotalRowMisses, negTotalColMisses


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
            print("Positive model: %d ngrams -> %d words" % (mc.pos_model.transCountMatrix.shape[0], mc.pos_model.transCountMatrix.shape[1]))
            print("Negative model: %d ngrams -> %d words" % (mc.neg_model.transCountMatrix.shape[0], mc.neg_model.transCountMatrix.shape[1]))
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
