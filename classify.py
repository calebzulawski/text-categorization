#!/usr/bin/env python3
import corpora
import tc
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('training')
parser.add_argument('testing')
parser.add_argument('predictions')
parser.add_argument('-s', '--stopwords', default='stopwords.txt')
args = parser.parse_args()

# Load the classifier object
classifier = tc.Classifier();
print('Loading stopwords')
classifier.load_stop_words(args.stopwords)

# Load the training data
print('Loading document labels')
directory, files = corpora.load_labeled_corpus_file(args.training)

# Calculate statistics and probabilities
print('Calculating document statistics')
countByDoc, countByClass, vocabulary = classifier.load_corpus_statistics(directory, files)
print('Calculating document probabilities')
prior, conditional = classifier.calculate_probabilities(vocabulary, countByDoc, countByClass, files)

# Load training data and classify
print('Classify test documents')
directory, testfiles = corpora.load_corpus_file(args.testing)
labels = classifier.classify(directory, testfiles, prior, conditional)

# Write labels to file
corpora.write_labeled_corpus_file(args.predictions, testfiles, labels)
print('Predictions written to ' + args.predictions)
