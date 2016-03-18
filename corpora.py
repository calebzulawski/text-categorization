import os
import random

def load_labeled_corpus_file(filename):
    labels = {}
    directory = os.path.dirname(filename)
    for line in open(filename, 'r'):
        subfilename = line.split()[0]
        labels[subfilename] = line.split()[1]
    return (directory, labels)

def load_kfold_corpus(filename, k):
    directory, labels = load_labeled_corpus_file(filename)
    shuffled = list(labels.items())
    random.shuffle(shuffled)
    folds = [({}, []) for i in range(k)]
    i = 0;
    for d, c in shuffled:
        for j in range(k):
            if i == j:
                folds[j][1].append(d)
            else:
                folds[j][0][d] = c
        i = (i + 1) % k
    return (directory, folds)

def load_corpus_file(filename):
    files = []
    directory = os.path.dirname(filename)
    for line in open(filename, 'r'):
        files.append(line.rstrip())
    return (directory, files)

def write_labeled_corpus_file(filename, documents, labels):
    if len(documents) != len(labels):
        raise ValueError('Number of labels does not match number of documents!')
    with open(filename, 'w') as f:
        for i in range(len(documents)):
            f.write(documents[i] + ' ' + labels[i] + '\n')
