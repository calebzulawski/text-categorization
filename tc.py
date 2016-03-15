import nltk
import re
import scipy
import numpy
import os

class Classifier():
    def __init__(self):
        self.stopwords = []
        self.stemmer = nltk.stem.snowball.SnowballStemmer('english')

    def load_stop_words(self, filename):
        with open(filename, 'r') as f:
            self.stopwords = f.read().split()
            return self

    def __statistics__(self, directory, filename):
        with open(os.path.join(directory, filename), 'r') as f:
            data = f.read().lower()
            frequency = {}
            for sent in nltk.sent_tokenize(data):
                for word in nltk.word_tokenize(sent):
                    if word not in self.stopwords and re.search('[a-z0-9]', word):
                        stemmed = self.stemmer.stem(word)
                        if stemmed in frequency:
                            frequency[stemmed] += 1
                        else:
                            frequency[stemmed] = 1
            return frequency

    def load_corpus_statistics(self, directory, documents):
        # Calculate frequencies per document
        f = {}
        terms = []
        for document in documents:
            f[document] = self.__statistics__(directory, document)
            for term in f[document]:
                if term not in terms:
                    terms.append(term)

        return (f, terms)

    def calculate_probabilities(self, terms, frequencies, labels):
        documents = list(frequencies.keys())
        classes = set(labels.values())

        prior = {}
        probabilities = {}

        for c in classes:
            prior[c] = list(labels.values()).count(c)/len(list(labels.values()))

        for c in classes:
            # Calculate denominator
            denom = len(terms); # smoothing
            classdocs = [d for d in documents if labels[d] == c]
            for d in classdocs:
                denom += numpy.sum(list(frequencies[d].values()))

            # Calculate probabilities
            probabilities[c] = {}

            for d in classdocs:
                for term in frequencies[d]:
                    if term in probabilities[c]:
                        probabilities[c][term] += frequencies[d][term]/denom
                    else:
                        probabilities[c][term] = (1 + frequencies[d][term])/denom # smoothing

        return (prior, probabilities)

    def classify(self, directory, documents, prior, probabilities):
        predictions = []
        for document in documents:
            f = self.__statistics__(directory, document)
            maxClass = None
            maxProb = float('-inf')
            for c in probabilities:
                prob = numpy.log(prior[c])
                for term in f:
                    if term in probabilities[c]:
                        prob += numpy.log(probabilities[c][term]) * f[term]
                if prob > maxProb:
                    print(prob)
                    maxClass = c
                    maxProb = prob
            predictions.append(maxClass)
        return predictions
