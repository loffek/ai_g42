
class CorpusReader():
    def __init__(self, filename):
        self.filename = filename

    def reviews(self):
        with open(self.filename, 'r', encoding='latin1') as f:
            for line in f:
                yield line.rstrip('\n')

