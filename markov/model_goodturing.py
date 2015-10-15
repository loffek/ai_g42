from .model import MarkovModel

from corpus import CorpusReader
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
import nltk
import time

from libs import sgts

PAD_TOKEN = "_"

class MarkovModelGoodTuring(MarkovModel):

    def __init__(self, order):
        self.k = order
        self.transCountMatrix = None
        self.transProbMatrix = None

        self.ngramHash = {} # maps an ngram to its row index in self.transCountMatrix
        self.wordHash = {} # maps a word to its col index in self.transCountMatrix

    def _tokenize(self, text):
        tokens = nltk.word_tokenize(text)
        if self.k == 0:
            tokens = tokens + [PAD_TOKEN] # add only the stop token
        else:
            tokens = [PAD_TOKEN]*(self.k) + tokens + [PAD_TOKEN]*(self.k) # add start and stop tokens

        return tokens

    def getProb(self, review, debuginfo={}):
        debuginfo.update({
            'totalRowMisses': 0,    # number of prevstates we didn't know
            'totalColMisses': 0,    # number of words we haven't seen before
            'totalTransMisses': 0,  # number of transistions we haven't seen before
            'transProbs': [],       # the probability of each transition
        })

        tokens = self._tokenize(review)
        ngrams = list(nltk.ngrams(tokens, self.k)) # this not effective, but works

        words = tokens[self.k:] # skip the first padding

        totalProb = 1.0
        for i, word in enumerate(words):
            prevstates = ngrams[i]
            transDebug = {}
            transProb = self.getTransitionProb(prevstates, word, transDebug)

            #multiply the totalProb
            totalProb *= transProb

            debuginfo['totalRowMisses'] += transDebug['rowmiss']
            debuginfo['totalColMisses'] += transDebug['colmiss']
            debuginfo['totalTransMisses'] += transDebug['transmiss']
            debuginfo['transProbs'].append(transDebug)

        return totalProb

    def getTransitionProb(self, prevstates, word, debuginfo):
        debuginfo.update({
            'from'      : prevstates,
            'to'        : word,
            'prob'      : 0,
            'count'     : 0,
            'rowmiss'   : 0,     #we didn't know these prevstates
            'colmiss'   : 0,     #we haven't seen this word before
            'transmiss' : 0,     #we knew the prevstate and the word, but we haven't a transition between them
        })

        if self.k == 0:
            row = 0 # just the only row we've got
        else:
            row = self.ngramHash.get(prevstates, None)
        col = self.wordHash.get(word, None)

        probSmooth = 0
        count = 0
        if row is None:
            debuginfo['rowmiss'] = 1
            if col is None:
                debuginfo['colmiss'] = 1
                raise Exception("What to do here?")
            else:
                countSmooth = 1
                raise Exception("What to do here?")

        else:
            if col is None:
                debuginfo['colmiss'] = 1
                col = self.transProbMatrix.shape[1]-1 # the last column is for the unknown word
                count = 0
            else:
                count = self.transCountMatrix[row, col]

            probSmooth = self.transProbMatrix[row, col]
            if probSmooth == 0:
                debuginfo['transmiss'] = 1

                col = self.transProbMatrix.shape[1]-1 # the last column is for unknown transitions as well
                probSmooth = self.transProbMatrix[row, col]

        debuginfo['count'] = count
        debuginfo['prob'] = probSmooth

        return probSmooth

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
        self.transProbMatrix = lil_matrix((ngramCounter, wordCounter+1), dtype=np.float64)

        revno = 1
        for review in reader.reviews():
            print("%4d" %(revno))

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
            revno += 1

        # convert it to compact csr_matrix format!
        self.transCountMatrix = csr_matrix(self.transCountMatrix)

        counts = {}
        for ngram, row in self.ngramHash.items():
            for word, col in self.wordHash.items():
                counts[word] = self.transCountMatrix[row, col]

            tic = time.time()
            probs, p0 = sgts.simpleGoodTuringProbs(counts)
            for word, col in self.wordHash.items():
                self.transProbMatrix[row, col] = probs[word]
            self.transProbMatrix[row, -1] = p0 # add p0 to the last column!
            toc = time.time()
            print("Elapsed: %.2f s" % (toc-tic))

            #probs["*unknown*"] = p0
            #highscore = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            #totalProb = 0
            #for (word, prob) in highscore:
            #    print("%20s : %4d , P = %.4f" % (word, counts.get(word, 0), prob))
            #    totalProb += prob
            #print("TOTAL: %.4f" % totalProb)


        #print("rows: %d, cols: %d" % (self.transCountMatrix.shape[0], self.transCountMatrix.shape[1]))

    def debugMatrix(self):
        print("%15s | " % (""), end="")
        for word, index in self.wordHash.items():
            print("%15s |" % word, end="")
        print()

        if self.k == 0:
            row = 0
            print("%15s | " % (""), end="")
            for word, col in self.wordHash.items():
                count = self.self.transCountMatrix[row, col]
                if count == 0:
                    print("%15s |" % (""), end="")
                else:
                    print("%15d |" % (count), end="")
            print()
        else:
            for ngram, row in self.ngramHash.items():
                print("%15s | " % (ngram,), end="")
                for word, col in self.wordHash.items():
                    count = self.self.transCountMatrix[row, col]
                    if count == 0:
                        print("%15s |" % (""), end="")
                    else:
                        print("%15d |" % (count), end="")
                        if count > 0:
                            count = ""
                print()

