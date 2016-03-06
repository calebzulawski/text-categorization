import nltk
import re

class Tc():
    def __init__(self):
        self.stopwords = []
        self.stemmer = nltk.PorterStemmer()

    def load_stop_words(self, filename):
        with open(filename, 'r') as f:
            self.stopwords = f.read().split()
            return self

    def load_file_as_tokens(self, filename):
        with open(filename, 'r') as f:
            data = f.read().lower()
            tokens = []
            for sent in nltk.sent_tokenize(data):
                for word in nltk.word_tokenize(sent):
                    if word not in self.stopwords and re.search('[a-z0-9]', word):
                        tokens.append(self.stemmer.stem(word));

            return tokens
