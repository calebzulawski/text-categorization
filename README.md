# text-categorization
Text categorization for ECE467

## Classifier information
The classifier is a Naive Bayes classifier

### Smoothing
The smoothing was done with add-one (Laplace) smoothing.  Other smoothing techniques would likely further improve performance. If I had implemented n-grams, I would have gone with Kneser-Ney smoothing.

### Tokenizer
The tokenizer used several techniques to improve performance.

1. The document is converted to all lowercase letters.  This improves performance negligibly, but makes later processing slightly easier.
2. The tokenizer uses the NLTK Punkt sentence tokenizer to split the document into sentences, then the NLTK Punkt word tokenizer to split that sentence into words.
3. The tokens are checked against a stop word list, and removed if present.  The stop words list is from http://www.ranks.nl/stopwords.
4. Tokens that do not contain any letters (i.e. contains only symbols and/or numbers) are removed.  This may not be beneficial for all corpora, but appears to work well in this domain.
5. Stemmers and lemmatizers were both tested, and all increased performance by small amounts.  The NLTK WordNet lemmatizer implementation was selected as it appeared to perform slightly more on the second provided corpus.

## Cross-validation
Cross-validation was done with k-folding, generally with k = 10.  The permutation of the document list was randomly generated, and split into 10 parts.  The sets were evaluated, trained against the other sets.  All of the predicted labels were then stored, and could be compared to the original labels.
