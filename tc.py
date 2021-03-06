import nltk
import re
import scipy
import numpy
import os

class Classifier():
    def __init__(self):
        self.stopwords = []
        self.stemmer = nltk.stem.snowball.SnowballStemmer('english')
        self.lemmatizer = nltk.stem.WordNetLemmatizer()

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
                    if word not in self.stopwords and re.search('[a-z]', word):
                        lemmatized = self.lemmatizer.lemmatize(word)
                        if lemmatized in frequency:
                            frequency[lemmatized] += 1
                        else:
                            frequency[lemmatized] = 1
            return frequency

    def load_corpus_statistics(self, directory, labeledDocuments):
        # Calculate frequencies per document
        countByDoc = {}
        countByClass = {}
        vocabulary = []
        for document, label in labeledDocuments.items():
            if label not in countByClass:
                countByClass[label] = {}

            countByDoc[document] = self.__statistics__(directory, document)

            for term, count in countByDoc[document].items():
                if term not in vocabulary:
                    vocabulary.append(term)
                if term not in countByClass[label]:
                    countByClass[label][term] = count;
                else:
                    countByClass[label][term] += count;

        return (countByDoc, countByClass, vocabulary)

    def calculate_probabilities(self, vocabulary, countByDoc, countByClass, labeledDocuments):
        documents = list(labeledDocuments.keys())
        classes = set(labeledDocuments.values())

        prior = {}
        conditional = {}

        for c in classes:
            conditional[c] = {}

            # Count documents in class
            Nc = list(labeledDocuments.values()).count(c)

            # Calculate prior for class
            prior[c] = Nc/len(documents)

            # Calculate conditional term probabilities
            denominator = numpy.sum(list(countByClass[c].values())) + len(vocabulary)
            for t in vocabulary:
                if t in countByClass[c]:
                    conditional[c][t] = (countByClass[c][t] + 1) / denominator
                else:
                    conditional[c][t] = 1 / denominator

        return (prior, conditional)

    def classify(self, directory, documents, prior, conditional):
        predictions = []
        for document in documents:
            f = self.__statistics__(directory, document)
            maxClass = None
            maxProb = float('-inf')
            for c in prior:
                prob = numpy.log(prior[c])
                for term in f:
                    if term in conditional[c]:
                        prob += numpy.log(conditional[c][term]) * f[term]
                if prob > maxProb:
                    maxClass = c
                    maxProb = prob
            predictions.append(maxClass)
        return predictions
