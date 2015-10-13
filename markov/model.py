from corpus import CorpusReader
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
import nltk

PAD_TOKEN = "_"

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
        tokens = [PAD_TOKEN]*(self.k) + tokens + [PAD_TOKEN]*(self.k) # add start and stop tokens, # TODO: add extra stop token? so the stop char is the last emitted state?
        return tokens

    def _getNGrams(self, tokens, n):
        return ngrams

    def getProb(self, review):
        totalRowMisses = 0 # number of prevstates we didn't know
        totalColMisses = 0 # number of words we haven't seen before

        tokens = self._tokenize(review)
        ngrams = list(nltk.ngrams(tokens, self.k)) # this not effective, but works

        words = tokens[self.k:] # skip the first padding

        totalProb = 1.0
        for i, word in enumerate(words):
            prevstates = ngrams[i]
            transProb, rowmiss, colmiss = self.getTransitionProb(prevstates, word)

            #multiply the totalProb
            totalProb *= transProb
            totalRowMisses += rowmiss
            totalColMisses += colmiss
        return totalProb, totalRowMisses, totalColMisses

    def getTransitionProb(self, prevstates, word):
        if self.smoothing != 'laplace':
            raise Exception("UNKNOWN SMOOTHING!")

        rowmiss = 0 #we didn't know these prevstates
        colmiss = 0 #we haven't seen this word before

        numCols = self.transCountMatrix.shape[1]

        if self.k == 0:
            row = 0 # just the only row we've got
        else:
            row = self.ngramHash.get(prevstates, None)
        col = self.wordHash.get(word, None)

        rowSumSmooth = numCols + 1 # add 1 for each word and 1 for the *unknown* word
        if row is None:
            rowmiss = 1
            # we don't know the prev state...
            # (keep default rowSumSmooth)
            if col is None:
                colmiss = 1
                # ...and it is the *unknown* word
                countSmooth = 1
                #print("ROW AND COL MISS!")
            else:
                # ...but we know the word (not that it matters)
                countSmooth = 1
                #print("ROW MISS!")

        else:
            # we know the prev states...
            rowSum = self.transCountMatrix.sum(axis=1)[row]
            rowSumSmooth += rowSum # add it to the default smooth value

            if col is None:
                colmiss = 1
                # ...but not the new word
                countSmooth = 1
                #print("COL MISS!")
            else:
                # everything ok
                countSmooth = self.transCountMatrix[row, col] + 1

        Ptrans = countSmooth / rowSumSmooth

        #if self.k == 0:
        #    print("'%s' %d / %d = %.4f" % (word, countSmooth, rowSumSmooth, Ptrans))
        #else:
        #    print("%s -> '%s' %d / %d = %.4f" % (prevstates, word, countSmooth, rowSumSmooth, Ptrans))

        return Ptrans, rowmiss, colmiss

    def trainOnCorpus(self, reviewfile):
        reader = CorpusReader(reviewfile)

        # count ngrams (prev states) and words (words/current state)
        # and create hashtables
        ngramCounter = 0
        wordCounter = 0

        for review in reader.reviews():
            #print(review)
            tokens = self._tokenize(review)

            if (self.k == 0):
                # in this case, make sure we get 1 row in the transitionMatrix
                self.ngramHash[(PAD_TOKEN,)] = 0
                ngramCounter = 1
            else:
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
            words = tokens[self.k:] # skip the first padding

            if self.k == 0:
                for word in words:
                    row = 0 # use the only row we got
                    col = self.wordHash[word]
                    self.transCountMatrix[row, col] += 1
            else:
                ngrams = list(nltk.ngrams(tokens, self.k)) # this not effective, but works

                for i, word in enumerate(words):
                    prevstates = ngrams[i]
                    row = self.ngramHash[prevstates]
                    col = self.wordHash[word]
                    self.transCountMatrix[row, col] += 1

        # convert it to compact csr_matrix format!
        self.transCountMatrix = csr_matrix(self.transCountMatrix)

        #print("rows: %d, cols: %d" % (self.transCountMatrix.shape[0], self.transCountMatrix.shape[1]))
