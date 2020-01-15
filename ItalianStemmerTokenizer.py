# HERE THE STEMMER CLASS
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import ItalianStemmer
import re
import string


class ItalianStemmerTokenizer(object):
    def __init__(self):
        self.stemmer = ItalianStemmer()

    def __call__(self, document):
        lemmas = []
        re_digit = re.compile('[0-9]')
        re_no_space = re.compile('[.;:!?,\"()\[\]]')
        re_space = re.compile('(<br\s*/><br\s*/>)|(\-)|(\/)')

        for t in word_tokenize(document):

            t = re_digit.sub(" ", t)
            t = re_space.sub(" ", t)
            t = re_no_space.sub("", t.lower())
            t = t.strip()  # it remove spaces before and after the characters

            if t != '':
                lemma = self.stemmer.stem(t)  # apply the stemmer
                lemmas.append(lemma)

        return lemmas