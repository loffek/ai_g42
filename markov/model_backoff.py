from .model import MarkovModel
from .model_laplace import MarkovModelLaplace
import nltk

# See http://courses.washington.edu/ling570/gina_fall11/slides/ling570_class8_smoothing.pdf
# for details about smoothing

class MarkovModelBackoff(MarkovModel):

    def __init__(self, order):
        self.k = order
        self.models = []
        ## TODO: use 0-order as base (the source above calls this 'unigram')
        for k in range(0, self.k+1):
            self.models.append(MarkovModelLaplace(k))

    def trainOnCorpus(self, file):
        for model in self.models:
            model.trainOnCorpus(file)

    def getProb(self, review, debuginfo={}):

        debuginfo.update({
            'totalRowMisses': 0,    # number of prevstates we didn't know
            'totalColMisses': 0,    # number of words we haven't seen before
            'totalTransMisses': 0,  # number of transistions we haven't seen before
            'transProbs': [],       # the probability of each transition
        })

        tokens = self.models[self.k]._tokenize(review)
        ngrams = list(nltk.ngrams(tokens, self.k)) # this not effective, but works

        words = tokens[self.k:] # skip the first padding

        totalProb = 1.0
        for i, word in enumerate(words):
            prevstates = ngrams[i]

            k = self.k # always start at max k
            alpha = 1 # backoff punishment

            miss = True # dummy to get inside loop
            while miss and k >= 0:

                transDebug = {}
                # alpha = 1, 0.4, 0.16, ...
                transProb = alpha * self.models[k].getTransitionProb(prevstates, word, transDebug)

                debuginfo['transProbs'].append(transDebug)

                miss = (transDebug['rowmiss'] or transDebug['colmiss'] or transDebug['transmiss'])
                k -= 1 #reduce k for each iteration
                alpha = alpha * 0.1 # increase the punishment for each iteration
                prevstates = prevstates[1:] # and reduce the prevstates

            #multiply the totalProb
            totalProb *= transProb

            # add the misses (only applies to the base-case 0-order model)
            debuginfo['totalRowMisses'] += transDebug['rowmiss']
            debuginfo['totalColMisses'] += transDebug['colmiss']
            debuginfo['totalTransMisses'] += transDebug['transmiss']
        return totalProb

