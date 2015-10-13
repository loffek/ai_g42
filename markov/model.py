
## Superclass for smoothed models

class MarkovModel():
    def __init__(self):
        pass

    def trainOnCorpus(self, file):
        self.smoothed_model.trainOnCorpus(file)

    def getProb(self, text):
        return self.smoothed_model.getProb(text)
