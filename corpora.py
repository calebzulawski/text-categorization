import os

def load_labeled_corpus_file(filename):
    labels = {}
    directory = os.path.dirname(filename)
    for line in open(filename, 'r'):
        subfilename = line.split()[0]
        labels[subfilename] = line.split()[1]
    return (directory, labels)

def load_corpus_file(filename):
    files = []
    directory = os.path.dirname(filename)
    for line in open(filename, 'r'):
        files.append(line)
    return (directory, files)

def write_labeled_corpus_file(filename, documents, labels):
    if len(documents) != len(labels):
        raise ValueError('Number of labels does not match number of documents!')
    with open(filename, 'w') as f:
        for i in range(len(documents)):
            f.write(documents[i] + ' ' + labels[i] + '\n')
