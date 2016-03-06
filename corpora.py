import os

def load_corpus_file_labels(filename):
    labels = {}
    for line in open(filename, 'r'):
        labels[os.path.abspath(line.split()[0])] = line.split()[1]
    return labels
