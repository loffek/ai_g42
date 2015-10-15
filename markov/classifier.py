import pickle

from .model_laplace import MarkovModelLaplace
from .model_backoff import MarkovModelBackoff
from .model_goodturing import MarkovModelGoodTuring

from constants import SENTIMENT

class MarkovClassifier:
    def __init__(self, order, smoothing):
        self.k = order
        self.smoothing = smoothing
        if smoothing == 'laplace':
            self.pos_model = MarkovModelLaplace(self.k)
            self.neg_model = MarkovModelLaplace(self.k)
        elif smoothing == 'backoff':
            self.pos_model = MarkovModelBackoff(self.k)
            self.neg_model = MarkovModelBackoff(self.k)
        elif smoothing == 'sgts':
            self.pos_model = MarkovModelGoodTuring(self.k)
            self.neg_model = MarkovModelGoodTuring(self.k)
        else:
            raise Exception('unsupported smoothing')

    def trainOnCorpora(self, posfile, negfile):
        self.pos_model.trainOnCorpus(posfile)
        self.neg_model.trainOnCorpus(negfile)
        return 0

    def printDebug(self, debugInfo):
        print("Total Col Misses: %d" % debugInfo['totalColMisses'])
        print("Total Row Misses: %d" % debugInfo['totalRowMisses'])
        print("Total Trans Misses: %d" % debugInfo['totalTransMisses'])
        for trans in debugInfo['transProbs']:
            if trans['colmiss']:
                if trans['rowmiss']:
                    print("P(\033[31m%60s\033[0m -> \033[31m%20s\033[0m) = %.2e (%d)" % (trans['from'], trans['to'], trans['prob'], trans['count']))
                else:
                    print("P(\033[32m%60s\033[0m -> \033[31m%20s\033[0m) = %.2e (%d)" % (trans['from'], trans['to'], trans['prob'], trans['count']))
            else:
                if trans['rowmiss']:
                    print("P(\033[31m%60s\033[0m -> \033[32m%20s\033[0m) = %.2e (%d)" % (trans['from'], trans['to'], trans['prob'], trans['count']))
                else:
                    if trans['transmiss']:
                        print("P(\033[93m%60s\033[0m -> \033[93m%20s\033[0m) = %.2e (%d)" % (trans['from'], trans['to'], trans['prob'], trans['count']))
                    else:
                        print("P(\033[32m%60s\033[0m -> \033[32m%20s\033[0m) = %.2e (%d)" % (trans['from'], trans['to'], trans['prob'], trans['count']))


    def classify(self, text, debug=False, debugInfo={}):
        posDebugInfo = {}
        pos_likelihood = self.pos_model.getProb(text, posDebugInfo)
        negDebugInfo = {}
        neg_likelihood = self.neg_model.getProb(text, negDebugInfo)

        if debug:
            print()
            print("---- DEBUG INFO ----")
            print("review: ")
            print(text)
            print("POSITIVE MODEL:")
            print("P = %.4e" % pos_likelihood)
            self.printDebug(posDebugInfo)

            print()
            print("NEGATIVE MODEL:")
            print("P = %.4e" % neg_likelihood)
            self.printDebug(negDebugInfo)

        debugInfo.update({
            'pos': posDebugInfo,
            'neg': negDebugInfo,
        })

        if pos_likelihood > neg_likelihood:
            return SENTIMENT.POSITIVE
        elif neg_likelihood > pos_likelihood:
            return SENTIMENT.NEGATIVE
        return SENTIMENT.NEUTRAL


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
            if mc.smoothing == 'laplace':
                print("Laplace classifier")
                print("Positive model: %d ngrams -> %d words" % (mc.pos_model.transCountMatrix.shape[0], mc.pos_model.transCountMatrix.shape[1]))
                print("Negative model: %d ngrams -> %d words" % (mc.neg_model.transCountMatrix.shape[0], mc.neg_model.transCountMatrix.shape[1]))
            elif mc.smoothing == 'backoff':
                print("Backoff classifier")

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
