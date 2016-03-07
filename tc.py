import nltk
import re
import scipy
import numpy
import os

class Frequencies():
    def __init__(self, terms, documents, matrix):
        self.terms = terms
        self.documents = documents
        self.matrix = matrix
        self.labels = {}
        self.classes = {}

    def set_labels(self, labels):
        self.labels = labels
        self.classes = list(labels.keys())

    def get(self, t, d):
        if t not in self.terms:
            return 0
        if d not in self.documents:
            return 0
        else:
            i = self.terms.index(t)
            j = self.classes.index(c)
            return self.matrix[i,j]

class Weights():
    def __init__(self, terms, classes, matrix):
        self.terms = terms
        self.classes = classes
        self.matrix = matrix

    def get(self, c, t):
        if t not in self.terms:
            return 0
        if c not in self.classes:
            return 0
        else:
            i = self.terms.index(t)
            j = self.classes.index(c)
            return self.matrix[i,j]

class Classifier():
    def __init__(self):
        self.stopwords = []
        self.stemmer = nltk.PorterStemmer()

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
                        if word in frequency:
                            frequency[word] += 1
                        else:
                            frequency[word] = 1
            return frequency

    def load_corpus_statistics(self, directory, docs):
        if type(docs) is list:
            documents = docs
            labels = {}
        elif type(docs) is dict:
            documents = list(docs.keys())
            labels = docs
        else:
            raise TypeError("Input must be list of filenames or a map of filenames to classes")

        terms = []

        print('Calculating term frequencies')
        # Calculate frequencies per document
        d = {}
        for document in documents:
            d[document] = self.__statistics__(directory, document)
            for term in d[document]:
                if term not in terms:
                    terms.append(term)

        # Generate sparse frequency matrix
        matrix = scipy.sparse.lil_matrix((len(terms), len(documents)))

        for i in range(len(terms)):
            term = terms[i]
            for j in range(len(documents)):
                doc = documents[j]
                if terms[i] in d[documents[j]]:
                    matrix[i,j] = d[doc][term]

        f = Frequencies(terms, documents, matrix)
        f.set_labels(labels)

        return f

    def calculate_weights(self, stats):
        # Since sparsity won't change, convert to CSR for faster operations
        matrix = stats.matrix.tocsr()
        terms = stats.terms
        documents = stats.documents
        labels = stats.labels
        classes = stats.classes

        # TF transform
        print('Performing TF transform')
        matrix = numpy.log1p(matrix)

        # IDF transform
        print('Performing IDF transform')
        for i in range(len(terms)):
            termcount = numpy.count_nonzero(matrix[i,:])
            for j in matrix[i,:].nonzero()[1]:
                matrix[i,j] *= (len(documents) / termcount)

        # length norm
        print('Normalizing tf-idf by L^2 norm')
        for j in range(len(documents)):
            norm = numpy.sqrt(matrix[:,j].multiply(matrix[:,j]).sum(0))
            for i in matrix[:,j].nonzero()[0]:
                matrix[i,j] /= norm

        # calculate weights
        print('Calculating weights')
        weights = numpy.zeros((len(terms), len(classes)))
        for j in range(len(classes)):
            c = classes[j]
            js = [documents.index(x) for x in labels if labels[x] != c]
            denom = matrix[:,js].sum()
            for i in range(len(terms)):
                weights[i,j] = numpy.log((matrix[i,js].sum()+1)/(denom + len(terms)))

        # normalize weights
        print('Normalizing weights')
        for j in range(len(classes)):
            weights[:,j] /= numpy.sum(weights[:,j])

        return Weights(terms, classes, matrix)

    def classify(self, weights, directory, documents):
        print('Classifying documents')
        labels = []
        stats = self.load_corpus_statistics(directory, documents)
        for document in documents:
            best = float('inf')
            bestclass = None
            for c in weights.classes:
                total = 0
                for term in stats.terms:
                    total += stats.get(term, document) * weights.get(term, c)
                if total < best:
                    best = total
                    bestclass = c
            labels.append(best)
        return labels
