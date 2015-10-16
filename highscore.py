from markov.classifier import MarkovClassifier
import numpy as np
import math

NUM_SCORES = 20


def getHighScore(model):
    rowHash = {}
    for ngram, r in model.ngramHash.items():
        rowHash[r] = ngram

    colHash = {}
    for word, c in model.wordHash.items():
        colHash[c] = word

    ones = 0
    twices = 0
    total = 0
    unique = 0

    rows, cols = model.transCountMatrix.nonzero()
    for i in range(0,len(rows)):
        r = rows[i]
        c = cols[i]
        count = model.transCountMatrix[r, c]
        if count > 0:
            if count == 1:
                ones += 1
                if ones <= NUM_SCORES:
                    ngram = rowHash[r]
                    word = colHash[c]
                    print("%20s -> %20s: %4d" % (ngram, word, count))
            if count == 2:
                twices += 1
                if twices <= NUM_SCORES:
                    ngram = rowHash[r]
                    word = colHash[c]
                    print("%20s -> %20s: %4d" % (ngram, word, count))

            total += count
            unique += 1
    print("ONES  : %d" % ones)
    print("TWICES: %d" % twices)
    print("TOTAL : %d" % total)
    print("UNIQUE: %d" % unique)

    backup = []
    for place in range(0,NUM_SCORES):
        cols = model.transCountMatrix.shape[1]
        elem = np.argmax(model.transCountMatrix.todense())
        #print(model.transCountMatrix.max())
        row = int(math.floor(elem / cols))
        col = elem % cols

        count = model.transCountMatrix[row, col]
        if count == 1:
            break
        ngram = rowHash[row]
        word = colHash[col]

        backup.append({
            'r': row,
            'c':col,
            'count': count,
            'ngram': ngram,
            'word': word,
        })

        # for next max find
        model.transCountMatrix[row, col] = 0
    return backup

def printStats(backup, model):
    for item in backup:
        debugInfo = {}
        prob = model.getTransitionProb(item['ngram'], item['word'], debugInfo)
        print("%20s -> %20s: %4d, %.4f" % (item['ngram'], item['word'], item['count'], prob))


    # this data is for 0-order (positive) Markov models
    word = "21st"
    debugInfo = {}
    model.getTransitionProb("", word, debugInfo)
    print("%20s -> %20s: %4d, %.4e" % ("", word, debugInfo['count'], debugInfo['prob']))
    word = "jean-claud"
    debugInfo = {}
    model.getTransitionProb("", word, debugInfo)
    print("%20s -> %20s: %4d, %.4e" % ("", word, debugInfo['count'], debugInfo['prob']))

    # this data is for 0-order (negative) Markov models
    word = "graphic"
    debugInfo = {}
    model.getTransitionProb("", word, debugInfo)
    print("%20s -> %20s: %4d, %.4e" % ("", word, debugInfo['count'], debugInfo['prob']))
    word = "laddish"
    debugInfo = {}
    model.getTransitionProb("", word, debugInfo)
    print("%20s -> %20s: %4d, %.4e" % ("", word, debugInfo['count'], debugInfo['prob']))


    word = "?????????"
    debugInfo = {}
    model.getTransitionProb("", word, debugInfo)
    print("%20s -> %20s: %4d, %.4e" % ("", word, debugInfo['count'], debugInfo['prob']))





print("========= k0 laplace ==========")
mc = MarkovClassifier.loadFromFile('savefiles/k0laplaceAB')
print("POS MODEL")
posbak = getHighScore(mc.pos_model)
print()
print("NEG MODEL")
negbak = getHighScore(mc.neg_model)
print()


mc = MarkovClassifier.loadFromFile('savefiles/k0laplaceAB')
print("POS MODEL")
printStats(posbak, mc.pos_model)
print("NEG MODEL")
printStats(negbak, mc.neg_model)



print("============ k0 sgts ============")
mc = MarkovClassifier.loadFromFile('savefiles/k0sgtsAB')
print("POS MODEL")
posbak = getHighScore(mc.pos_model)
print()
print("NEG MODEL")
negbak = getHighScore(mc.neg_model)
print()


mc = MarkovClassifier.loadFromFile('savefiles/k0sgtsAB')
print("POS MODEL")
printStats(posbak, mc.pos_model)
print("NEG MODEL")
printStats(negbak, mc.neg_model)

#debuginfo = {}
#mc.pos_model.getTransitionProb(("_",), ".", debuginfo)
#print(debuginfo)
"""
from markov.model_laplace import MarkovModelLaplace

pos = MarkovModelLaplace(0)
pos.trainOnCorpus('data/corpora/training_3fold/posAB.txt')

sum = 0
for word, col in pos.wordHash.items():
    count = pos.transCountMatrix[0,col]
    sum += count
    print(count)

print(sum)

print("==============================")

sum = 0
for col in range(0, pos.transCountMatrix.shape[1]):
    count = pos.transCountMatrix[0,col]
    sum += count
    print(count)

print(sum)

print("==============================")


sum = 0
rows, cols = pos.transCountMatrix.nonzero()
for i in range(0,len(rows)):
    col = cols[i]
    count = pos.transCountMatrix[0,col]
    sum += count
    print(count)

print(sum)

print("==============================")

print(pos.transCountMatrix.sum(axis=1)[0])
print(pos.transCountMatrix.todense().sum(axis=1)[0])
"""
