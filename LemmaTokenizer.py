from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer


class LemmaTokenizer(object):
    def __init__(self):
        # Here the stemming operation for ex: 'dogs' -> 'dog'
        self.lemmatizer = WordNetLemmatizer()

    def __call__(self, document):
        lemmas = []
        for t in word_tokenize(document):
            # word_tokenize(document) allow to make a list with all the words occurring in the document
            # punctation not excluded !!
            # t = t.translate(table)#it remove punctation
            t = t.strip()  # it remove spaces before and after the characters
            lemma = self.lemmatizer.lemmatize(t)  # apply the lemmatization
            lemmas.append(lemma)
        return lemmas