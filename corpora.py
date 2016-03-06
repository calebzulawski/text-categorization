import os

def load_corpus_file_labels(filename):
    labels = {}
    directory = os.path.dirname(filename)
    for line in open(filename, 'r'):
        subfilename = os.path.abspath(os.path.join(directory, line.split()[0]))
        labels[subfilename] = line.split()[1]
    return labels
