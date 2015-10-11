from corpus import CorpusReader
import numpy as np
import re

WHITELIST=1
BLACKLIST=2

MODES = { \
WHITELIST : '[^\w\s-]', \
BLACKLIST : '[.,?!"\']' }


class MarkovModel():



    def __init__(self, order, smoothing,matching_mode=WHITELIST):
        self.k = order
        self.smoothing = smoothing
        self.set_of_words = set()
        self.regex = re.compile(MODES[matching_mode], re.U)

    def getProb(self, text,transitionMatrix,wordVector):
        filtered_text = self.filtering(text)
        prob = 0
        for word in filtered_text[1:len(filtered_text)]
                lastword = wordvector[wordVector.index(word) - 1]
                prob += transitionMatrix.item((wordVector.index(word),wordVector.index(lastword)))
        return prob

    def filtering(self,text):
        filtered_text = self.regex.sub(" ",text)
        return filtered_text

    def trainOnCorpus(self, reviewfile):
        reader = CorpusReader(reviewfile)
        wordVector=[]
        allWords = []
        for review in reader.reviews():
            filtered_text = self.filtering(review)
            list_of_words = filtered_text.split()
            for word in list_of_words:
                if word not in wordVector:
                    wordVector.append(word)
                allWords.append(word)          
        transitionMatrix = np.zeros([len(wordVector),len(wordVector)])
            #print("\"%s\"" % review)

        for word in allWords:
            nextWord= wordVector[wordVector.index(word)+1]
            transitionMatrix.item(wordVector.index(word),wordVector.index(nextWord) += 1

        for i in range(0,len(wordVector)):
            row = transitionMatrix[i] 
            transitionMatrix[i] = np.divide(row,row.sum())

        return transitionMatrix,wordVector

 #       def main(self):
 #           tranMatrix,wordVec = self.trainOnCorpus(text)
 #           return self.getProb(text,tranMatrix,wordVec)
