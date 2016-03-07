#!/usr/bin/env python3
import corpora
import tc

classifier = tc.Classifier();
classifier.load_stop_words('stopwords.txt')

directory, files = corpora.load_labeled_corpus_file('corpora/corpus1_train.labels')

stats = classifier.load_corpus_statistics(directory, files)

weights = classifier.calculate_weights(stats)

directory, testfiles = corpora.load_corpus_file('corpora/corpus1_test.list')

labels = classifier.classify(weights, directory, testfiles)

corpora.write_labeled_corpus_file('corpus1_test.predictions', testfiles, labels)
