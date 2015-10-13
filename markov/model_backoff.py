from .model import MarkovModel
from .model_laplace import MarkovModelLaplace

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

    def getProb(self, review):

        totalRowMisses = 0 # number of prevstates we didn't know
        totalColMisses = 0 # number of words we haven't seen before

        tokens = self.models[self.k]._tokenize(review)
        ngrams = list(nltk.ngrams(tokens, self.k)) # this not effective, but works

        words = tokens[self.k:] # skip the first padding

        totalProb = 1.0
        for i, word in enumerate(words):
            prevstates = ngrams[i]

            k = self.k # always start at max k

            rowmiss = 1 # dummy to get inside loop
            while (rowmiss > 0 or colmiss > 0) and k >= 0:
                if k != self.k:
                    if rowmiss:
                        print("rowmiss! reduce order to %d" % k)
                    elif colmiss:
                        print("colmiss! reduce order to %d (we can go to order 0 right away..." % k)

                transProb, rowmiss, colmiss = self.models[k].getTransitionProb(prevstates, word)
                k -= 1 #reduce k for each iteration

            #multiply the totalProb
            totalProb *= transProb

            # add the misses (only applies to the base-case 0-order model)
            totalRowMisses += rowmiss
            totalColMisses += colmiss
        return totalProb, totalRowMisses, totalColMisses

