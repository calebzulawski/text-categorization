#!/usr/bin/env python3
import corpora
import tc

files = corpora.load_corpus_file_labels('corpora/corpus1_test.labels')

c = tc.Tc();
c.load_stop_words('stopwords.txt')

c.calculate_weights(files)
