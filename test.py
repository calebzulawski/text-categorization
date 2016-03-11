#!/usr/bin/env python3
import corpora
import tc

classifier = tc.Classifier();
classifier.load_stop_words('stopwords.txt')

directory, files = corpora.load_labeled_corpus_file('corpora/corpus1_train.labels')

frequency, terms = classifier.load_corpus_statistics(directory, files)

weights = classifier.calculate_probabilities(terms, frequency, files)

directory, testfiles = corpora.load_corpus_file('corpora/corpus1_test.list')

labels = classifier.classify(directory, testfiles, probabilities)

corpora.write_labeled_corpus_file('corpus1_test.predictions', testfiles, labels)
