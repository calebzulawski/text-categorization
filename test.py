#!/usr/bin/env python3
import corpora
import tc

files = corpora.load_corpus_file_labels('corpora/corpus1_test.labels')

c = tc.Tc();
c.load_stop_words('stopwords.txt')

file1 = list(files.keys())[0]

print(c.load_file_as_tokens(file1))
