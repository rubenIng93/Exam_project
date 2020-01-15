from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import ItalianStemmer
import re
import string


class LemmaTokenizer(object):
    def __init__(self):
        # Here the stemming operation for ex: 'dogs' -> 'dog'
        self.lemmatizer = WordNetLemmatizer()

    def __call__(self, document):
        lemmas = []
        re_digit = re.compile("[0-9]")

        for t in word_tokenize(document):
            t = t.strip()  # it remove spaces before and after the characters
            lemma = self.lemmatizer.lemmatize(t)  # apply the lemmatization

            # Remove tokens with only punctuation chars and digits
            if lemma not in string.punctuation and len(lemma) > 3 and len(lemma) < 16 and not re_digit.match(lemma):
                lemmas.append(lemma)

        return lemmas