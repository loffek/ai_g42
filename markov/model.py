from corpus import CorpusReader
from scipy.sparse import csr_matrix
import nltk


class MarkovModel():

    def __init__(self, order, smoothing):
        self.k = order
        self.smoothing = smoothing
        self.set_of_words = set()
        self.transCountMatrix = None
        self.ngramHash = {}
        self.emissionHash = {}

    def _tokenize(self, text):
        tokens = nltk.word_tokenize(text)
        tokens = ['_']*(self.k-1) + tokens + ['_']*(self.k-1) # add start and stop tokens
        return tokens

    def _getNGrams(self, tokens, n):
        return ngrams

    def getProb(self, text):
        return 0

    def getTransitionProb(self, ngram, token):
        return 0

    def trainOnCorpus(self, reviewfile):
        reader = CorpusReader(reviewfile)

        # count ngrams (prev states) and emissions (words/current state)
        ngramCounter = 0
        emissionCounter = 0

        for review in reader.reviews():
            tokens = self._tokenize(review)
            ngrams = nltk.ngrams(tokens, self.k)

            for ngram in ngrams:
                if ngram not in self.ngramHash:
                    self.ngramHash[ngram] = ngramCounter
                    ngramCounter += 1

            for token in tokens:
                if token not in self.emissionHash:
                    self.emissionHash[token] = emissionCounter
                    emissionCounter += 1

        for review in reader.reviews()
