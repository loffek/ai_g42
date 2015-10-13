from corpus import CorpusReader
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
import nltk


class MarkovModel():

    def __init__(self, order, smoothing):
        self.k = order
        self.smoothing = smoothing
        self.set_of_words = set()
        self.transCountMatrix = None

        self.ngramHash = {} # maps an ngram to its row index in transCountMatrix
        self.wordHash = {} # maps a word to its col index in transCountMatrix

    def _tokenize(self, text):
        tokens = nltk.word_tokenize(text)
        tokens = ['_']*(self.k) + tokens + ['_']*(self.k) # add start and stop tokens, # TODO: add extra stop token? so the stop char is the last emitted state?
        return tokens

    def _getNGrams(self, tokens, n):
        return ngrams

    def getProb(self, review):
        tokens = self._tokenize(review)
        ngrams = list(nltk.ngrams(tokens, self.k)) # this not effective, but works

        words = tokens[self.k:] # skip the first padding

        totalProb = 1.0
        for i, word in enumerate(words):
            prevstates = ngrams[i]
            totalProb *= self.getTransitionProb(prevstates, word)
        return totalProb

    def getTransitionProb(self, prevstates, word):
        if self.smoothing != 'laplace':
            raise Exception("UNKNOWN SMOOTHING!")

        numCols = self.transCountMatrix.shape[1]

        row = self.ngramHash.get(prevstates, None)
        col = self.wordHash.get(word, None)

        rowSumSmooth = numCols + 1 # add 1 for each word and 1 for the *unknown* word
        if row is None:
            # we don't know the prev state...
            # (keep default rowSumSmooth)
            if col is None:
                # ...and it is the *unknown* word
                countSmooth = 1
            else:
                # ...but we know the word (not that it matters)
                countSmooth = 1

        else:
            # we know the prev states...
            rowSum = self.transCountMatrix.sum(axis=1)[row]
            rowSumSmooth += rowSum # add it to the default smooth value

            if col is None:
                # but not the new word
                countSmooth = 1

            else:
                # everything ok
                countSmooth = self.transCountMatrix[row, col] + 1

        Ptrans = countSmooth / rowSumSmooth

        #print("%s -> '%s' %d / %d = %.4f" % (prevstates, word, countSmooth, rowSumSmooth, Ptrans))

        return Ptrans

    def trainOnCorpus(self, reviewfile):
        reader = CorpusReader(reviewfile)

        # count ngrams (prev states) and words (words/current state)
        # and create hashtables
        ngramCounter = 0
        wordCounter = 0

        for review in reader.reviews():
            tokens = self._tokenize(review)
            ngrams = nltk.ngrams(tokens, self.k)

            for ngram in ngrams:
                if ngram not in self.ngramHash:
                    self.ngramHash[ngram] = ngramCounter
                    ngramCounter += 1

            for token in tokens:
                if token not in self.wordHash:
                    self.wordHash[token] = wordCounter
                    wordCounter += 1

        # create the transitionMatrix
        # use the lil_matrix format now for inserts and convert to more compact csr_matrix later
        self.transCountMatrix = lil_matrix((ngramCounter, wordCounter), dtype=np.uint16)

        revno = 1
        for review in reader.reviews():
            print("%4d" %(revno))
            revno += 1

            tokens = self._tokenize(review)

            ngrams = list(nltk.ngrams(tokens, self.k)) # this not effective, but works

            words = tokens[self.k:] # skip the first padding

            for i, word in enumerate(words):
                prevstates = ngrams[i]
                #print("%s -> %s" % (prevstates, word))
                row = self.ngramHash[prevstates]
                col = self.wordHash[word]
                #print("adding transition to (%d, %d)" % (row, col))
                self.transCountMatrix[row, col] += 1

        # convert it to compact csr_matrix format!
        self.transCountMatrix = csr_matrix(self.transCountMatrix)
        print(self.transCountMatrix.todense())
        print("rows: %d, cols: %d" % (self.transCountMatrix.shape[0], self.transCountMatrix.shape[1]))

