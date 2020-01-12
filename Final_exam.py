import string
import pandas as pd
import LemmaTokenizer
import csv
from stop_words import get_stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def load_data(file):
    df = pd.read_csv(file, sep=',')
    print(df.head())
    print(df.describe())
    return df


# i need to remove the punctuation that is useless for the analysis
def remove_punctuation(dataframe):
    table = str.maketrans('', '', string.punctuation)
    for review, index in zip(dataframe['text'], dataframe.index):
        review = review.translate(table)
        dataframe.iloc[index,  0] = review


def dump_to_file(filename, y_pred, dataset):
    with open(filename, mode="w", newline="") as csvfile:
        # Headers
        fieldnames = ['Id', 'Predicted']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for ids, cls in zip(dataset.keys(), y_pred):
            writer.writerow({'Id': str(ids), 'Predicted': str(cls)})


# now i can use the LemmaTokenizer class already used in the previous lab
# to make the reviews ready for the vectorizer.
# To make a good analysis i should remove all the common stopwords, to do this
# i can exploit the stop_words library



data = load_data('development.csv')
print('Before removing punctuation:')
print(data['text'][:3])
remove_punctuation(data)
print('After that:')
print(data['text'][:3])

lemma = LemmaTokenizer.LemmaTokenizer()
stop_words = get_stop_words('it')
stop_words.extend(('quantum', 'hotel'))
vectorizer = TfidfVectorizer(tokenizer=lemma, lowercase=True, min_df=0.04, stop_words = stop_words)
X_tfidf = vectorizer.fit_transform(data['text'])
print('tfidf shape:')
print(X_tfidf.shape)
labels = data['class']


X_train, X_test, y_train, y_test = train_test_split(X_tfidf,
                                                    labels, test_size=0.2)

clf = RandomForestClassifier(n_estimators=500, criterion='entropy', max_depth=20)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(f1_score(y_test, y_pred, average='weighted'))


