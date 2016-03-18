#!/usr/bin/env python3

import corpora
import tc

classifier = tc.Classifier();
classifier.load_stop_words('stopwords.txt')

directory, folds = corpora.load_kfold_corpus('corpora/corpus2_train.labels', 10)

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

corpora.write_labeled_corpus_file('corpus2_train.predictions', totalfiles, totallabels)
