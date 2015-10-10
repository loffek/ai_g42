from corpus import CorpusReader

class MarkovModel():

    def __init__(self, order, smoothing):
        self.k = order
        self.smoothing = smoothing

    def getProb(self, text):
        return 0

    def trainOnCorpus(self, reviewfile):
        reader = CorpusReader(reviewfile)
        for review in reader.reviews():
            print("\"%s\"" % review)

        return 0

