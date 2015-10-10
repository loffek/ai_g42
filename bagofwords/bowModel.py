import re

WHITELIST = 1
BLACKLIST = 2


MODES = { \
    WHITELIST : '[^\w\s-]', \
    BLACKLIST : '[.,?!"\']' }


class BowModel:
    def __init__(self, matching_mode=WHITELIST):
        self.set_of_words = set()
        self.regex = re.compile(MODES[matching_mode], re.U)

    def __loadWordsFromFile(self, file_name):        
        file = open(file_name,'r',encoding = 'utf-8')
        line = file.readline()
        set_of_words = set()
        while line:
            word = line.strip()
            set_of_words.add(word)
            line = file.readline()
        return set_of_words

    def populate(self,file):
        self.set_of_words = self.__loadWordsFromFile(file)

    def evaluate(self, text):
        filtered_text = self.regex.sub(" ",text)
        list_of_words = filtered_text.split()
        score = 0
        for word in list_of_words:
            if word.lower() in self.set_of_words:
                score += 1
        return score

    def evaluateFromText(self, text):
        text = f.read()
        f.close()
        evaluate(text)

    

        
