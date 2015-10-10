
import pickle

class MarkovClassifier:
    def __init__(self, order, smoothing):
        self.k = order
        self.smoothing = smoothing

    @staticmethod
    def loadFromBuffer(buffer):
        return pickle.loads(buffer)

    @staticmethod
    def loadFromFile(filepath):
        with open(filepath, 'r') as f:
            return loadFromBuffer(f.read())

    def saveToBuffer():
        return pickle.dumps(this)

    def saveToFile(filepath):
        with open(filepath, 'w') as f:
            f.write(saveToBuffer())

    def trainOnCorpora(self, pos, neg):

    def classify(text):
        return 0

