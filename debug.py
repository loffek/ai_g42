

"""
from markov.model_laplace import MarkovModelLaplace
neg = MarkovModelLaplace(1)
neg.trainOnCorpus('negdebug.txt')
neg.debugMatrix()


pos = MarkovModelLaplace(1)
pos.trainOnCorpus('posdebug.txt')
pos.debugMatrix()
"""

"""
from markov.classifier import MarkovClassifier

# TRAIN AND SAVE

mc = MarkovClassifier(order=2, smoothing='laplace')
mc.trainOnCorpora(posfile='posdebug.txt', negfile='negdebug.txt')

print("POS MODEL:")
mc.pos_model.debugMatrix()
print("NEG MODEL:")
mc.neg_model.debugMatrix()

mc.saveToFile('debugmodel')


# LOAD AND CLASSIFY
mc = MarkovClassifier.loadFromFile('debugmodel')
print("POS MODEL:")
mc.pos_model.debugMatrix()
print("NEG MODEL:")
mc.neg_model.debugMatrix()


review = "n1 n3 n4 n5 . "
sentiment = mc.classify(review, debug=True)
print(sentiment.name)

review = "n1 p1 n2 n4 . "
sentiment = mc.classify(review, debug=True)
print(sentiment.name)


from corpus import CorpusReader
negread = CorpusReader('negdebug.txt')
posread = CorpusReader('posdebug.txt')

for review in negread.reviews():
    print(review)

for review in posread.reviews():
    print(review)
"""

"""
from markov.classifier import MarkovClassifier
mc = MarkovClassifier.loadFromFile('debugmodel')

word = "'hannibal'"
print(mc.pos_model.wordHash[word])

word = "olvidar'"
print(mc.pos_model.wordHash[word])
"""

import nltk

tokens = nltk.word_tokenize(text)
