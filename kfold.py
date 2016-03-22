#!/usr/bin/env python3
import corpora
import tc
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('training')
parser.add_argument('predictions')
parser.add_argument('-k', default=10)
parser.add_argument('-s', '--stopwords', default='stopwords.txt')
args = parser.parse_args()

# Load the classifier object
classifier = tc.Classifier()
print('Loading stopwords')
classifier.load_stop_words(args.stopwords)

# Create k-folds
print('k-folding for k = ' + str(args.k))
directory, folds = corpora.load_kfold_corpus(args.training, args.k)

# Classifying
print('Predicting labels for each hold-out set')
totalfiles = []
totallabels = []

for data in folds:
    training = data[0]
    testing = data[1]
    countByDoc, countByClass, vocabulary = classifier.load_corpus_statistics(directory, training)
    prior, conditional = classifier.calculate_probabilities(vocabulary, countByDoc, countByClass, training)
    labels = classifier.classify(directory, testing, prior, conditional)
    totalfiles.extend(testing)
    totallabels.extend(labels)

corpora.write_labeled_corpus_file(args.predictions, totalfiles, totallabels)
print('Predictions written to ' + args.predictions)
