import nltk

def load_file_as_tokens(filename):
    with open(filename, 'r') as f:
        data = f.read()
        tokens = nltk.word_tokenize(data)
        return tokens
