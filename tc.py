import nltk
import re
import scipy
import numpy

class Tc():
    def __init__(self):
        self.stopwords = []
        self.stemmer = nltk.PorterStemmer()

    def load_stop_words(self, filename):
        with open(filename, 'r') as f:
            self.stopwords = f.read().split()
            return self

    def load_file_statistics(self, filename):
        with open(filename, 'r') as f:
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

    def calculate_weights(self, labels):
        filenames = list(labels.keys())
        numdocs = len(filenames)
        terms = []

        # Calculate frequencies per document
        print("Calculating term frequencies")
        d = {}
        for filename in filenames:
            d[filename] = self.load_file_statistics(filename)
            for term in d[filename]:
                if term not in terms:
                    terms.append(term)

        numterms = len(terms)

        # Generate sparse frequency matrix
        matrix = scipy.sparse.lil_matrix((numterms, numdocs))

        for i in range(numterms):
            term = terms[i]
            for j in range(numdocs):
                doc = filenames[j]
                if terms[i] in d[filenames[j]]:
                    matrix[i,j] = d[doc][term]

        # Since sparsity won't change, convert to CSR for faster operations
        matrix = matrix.tocsr()

        # TF transform
        print("Performing TF transform")
        matrix = numpy.log1p(matrix)

        # IDF transform
        print("Performing IDF transform")
        for i in range(numterms):
            termcount = numpy.count_nonzero(matrix[i,:])
            for j in matrix[i,:].nonzero()[1]:
                matrix[i,j] *= (numdocs / termcount)

        # length norm
        print("Normalizing by L^2 norm")
        for j in range(numdocs):
            norm = numpy.sqrt(matrix[:,j].multiply(matrix[:,j]).sum(0))
            for i in matrix[:,j].nonzero()[0]:
                matrix[i,j] /= norm

        # calculate weights
        print("Calculating weights")
        weights = {}
        classes = numpy.unique(list(labels.values()))
        numclasses = len(classes)
        weights = numpy.zeros((numterms, numclasses))
        for j in range(numclasses):
            c = classes[j]
            js = [filenames.index(x) for x in labels if labels[x] != c]
            denom = matrix[:,js].sum()
            for i in range(numterms):
                weights[i,j] = numpy.log((matrix[i,js].sum()+1)/(denom + numterms))

        # normalize weights
        print("Normalizing weights")
        for j in range(numclasses):
            weights[:,j] /= numpy.sum(weights[:,j])
