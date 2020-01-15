import pandas as pd
import re
import ItalianStemmerTokenizer
from nltk.stem.snowball import ItalianStemmer
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from stop_words import get_stop_words
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV



def balance_ds(data):
    df_major = data.loc[data['class'] == 'pos']
    df_minor = data.loc[data['class'] == 'neg']
    df_down_sample = resample(df_major, replace=False, n_samples=9222, random_state=42)
    balanced_df = pd.concat([df_down_sample, df_minor])
    return balanced_df


def plot_distribution(df):
    pos_reviews = df.loc[df['class'] == 'pos'].count()
    neg_reviews = df.loc[df['class'] == 'neg'].count()
    x = [1, 2]
    labels = ['pos', 'neg']
    fig,ax = plt.subplots(figsize=(5, 5))
    ax.bar(x, [pos_reviews[1], neg_reviews[1]], tick_label=labels, color=['green', 'crimson'])
    plt.ylabel('Number of Review')
    fig.savefig('Distribution.png')
    plt.show()


def clean_data(data):
    re_digit = re.compile("[0-9\']")
    re_no_space = re.compile("[.;:!?,\"()\[\]]")
    re_space = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
    data = [re_no_space.sub("", line.lower()) for line in data]
    data = [re_space.sub(" ", line) for line in data]
    data = [re_digit.sub(" ", line) for line in data]
    return data


def remove_stop_words(data):
    removed_stop_words = []
    stop_words = get_stop_words('it')
    for review in data:
        removed_stop_words.append(
            ' '.join([word for word in review.split()
                      if word not in stop_words])
        )
    return removed_stop_words


def get_stemmed_text(data):
    stem = ItalianStemmer()
    return [' '.join([stem.stem(word) for word in review.split()]) for review in data]


def print_stats(y_test, y_pred):
    print('Classification results:')
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(f1_score(y_test, y_pred, average='weighted'))


def load_data(file):
    df = pd.read_csv(file, sep=',')
    return df


def into_tfidf(data):
    # lemma = LemmaTokenizer.LemmaTokenizer()
    stem = ItalianStemmerTokenizer.ItalianStemmerTokenizer()
    stopwords = get_stop_words('it')
    stopwords.extend(['abbi', 'abbiam', 'adess', 'allor', 'ancor', 'avemm', 'avend',
                      'aver', 'avess', 'avesser', 'avessim', 'avest', 'avet', 'avev',
                      'avevam', 'avra', 'avrann', 'avre', 'avrebb', 'avrebber',
                      'avrem', 'avremm', 'avrest', 'avret', 'avro', 'avut', 'com',
                      'contr', 'dentr', 'ebber', 'eran', 'erav', 'eravam', 'essend',
                      'fac', 'facc', 'facess', 'facessim', 'facest', 'fann', 'far',
                      'fara', 'farann', 'farebb', 'farebber', 'farem', 'farest',
                      'fec', 'fecer', 'fin', 'foss', 'fosser', 'fossim', 'fost',
                      'fumm', 'fur', 'giu', 'hann', 'lor', 'nostr', 'perc', 'piu',
                      'poc', 'poch', 'qual', 'quant', 'quas', 'quell', 'quest',
                      'quind', 'sar', 'sara', 'sarann', 'sare', 'sarebb', 'sarebber',
                      'sarem', 'sarest', 'senz', 'siam', 'sian', 'siat', 'siet',
                      'son', 'sopr', 'sott', 'stand', 'stann', 'star', 'stara',
                      'starann', 'starebb', 'starebber', 'starem', 'starest',
                      'stav', 'stavam', 'stemm', 'stess', 'stesser', 'stessim',
                      'stest', 'stett', 'stetter', 'sti', 'stiam', 'tutt', 'vostr'])

    vectorizer = TfidfVectorizer(tokenizer=stem, strip_accents='ascii',
                                 stop_words=stopwords, ngram_range=(1, 2),
                                 binary=True)
    X_tfidf = vectorizer.fit_transform(data)
    return X_tfidf


def dim_reduction(tfidf, idx):
    svd = TruncatedSVD(n_components=idx, random_state=25)
    data = svd.fit_transform(tfidf)
    return data


def dump_to_file(filename, y_pred, dataset):
    with open(filename, mode="w", newline="") as csvfile:
        # Headers
        fieldnames = ['Id', 'Predicted']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for id, cls in zip(dataset.index.values.tolist(), y_pred):
            writer.writerow({'Id': str(id), 'Predicted': str(cls)})



# BELOW HERE THE PROGRAM:
start_time = time.time()
# Loading data:
dev_set = load_data('development.csv')
eva_set = load_data('evaluation.csv')
print(dev_set.head())
print(dev_set.describe())

# First data analysis
print('Plotting data Distribution...')
plot_distribution(dev_set)

# Data Preprocessing:
print('Start Preprocessing:')
# 1) Data Cleaning
# a. Removing punctuation, digits
print('Cleaning text...')
cleaned_dev = clean_data(dev_set['text'])
cleaned_eva = clean_data(eva_set['text'])
# b. Removing stop words
print('Removing stopwords...')
no_sw_dev = remove_stop_words(cleaned_dev)
no_sw_eva = remove_stop_words(cleaned_eva)
# c. Getting stemmed words
print('Applying Stemmer...')
stem_dev = get_stemmed_text(no_sw_dev)
stem_eva = get_stemmed_text(no_sw_eva)
# 2) Vectorization
print('Vectorization...')
stop_words = ['milano','un','tutti','ci','lo','era']
tfidf_vec = TfidfVectorizer(ngram_range=(1,2), binary=True, stop_words=stop_words)
X = tfidf_vec.fit_transform(cleaned_dev)
X_test = tfidf_vec.transform(cleaned_eva)

# Classification

print('Start Classification:')
labels = dev_set['class']
# 1) Split data in Training and test set
X_train, X_val, y_train, y_val = train_test_split(X, labels, test_size=0.2)
# 2) Define a classifier
clf = LinearSVC()
# clf = RandomForestClassifier(n_estimators=100, min_impurity_decrease=0.12)
#clf = GradientBoostingClassifier(n_estimators=250, min_impurity_decrease=0.2)
# clf = SVC(gamma='auto')
# clf = MultinomialNB()
# Performing the Grid Search Cross Valitaion
param_grid = {'C':[0.01, 0.05, 0.25, 0.5, 0.75, 1]}
gridsearch = GridSearchCV(clf, param_grid, scoring='f1_weighted', cv=5)
clf = gridsearch.fit(X_train, y_train)
print("Best model configuration is:")
print(clf.best_params_)
print("with f1 = ", clf.best_score_)
# 3) Output the statistics & perform the cross validation
print_stats(y_val, clf.predict(X_val))
f1_cv = cross_val_score(clf, X, labels, cv=3, scoring='f1_weighted')
mean_f1 = f1_cv.mean()
std_f1 = f1_cv.std()
print(f"f1 (statistics): {mean_f1:.2f} (+/- {std_f1:.2f})")
# 4) Fit the final model with the best hyperparameters found
print('Fitting final model...')
final_model = clf.best_estimator_
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)


# Dumping to file
print('Dumping in file...')
dump_to_file('result.csv', y_pred, eva_set)
print('Process done in %s seconds' % (time.time() - start_time))


