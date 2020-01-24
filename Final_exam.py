import pandas as pd
import seaborn as sns
import csv
import time
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import ItalianStemmer
import re
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from stop_words import get_stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


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
    data = [re_no_space.sub("", line.lower()) for line in data]
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


def get_top_terms(num, final_model, tfidf_vec):
    print('\nTOP ', num, ' Terms:')
    feature_to_coef = {
        word: coef for word, coef in zip(
            tfidf_vec.get_feature_names(), final_model.coef_[0]
        )
    }
    pos_terms = []
    print('Positive:')
    for best_positive in sorted(
            feature_to_coef.items(),
            key=lambda x: x[1],
            reverse=True)[:num]:
        print(best_positive)
        pos_terms.append(best_positive)

    print('Negative:')
    neg_terms = []
    for best_negative in sorted(
            feature_to_coef.items(),
            key=lambda x: x[1])[:num]:
        print(best_negative)
        neg_terms.append(best_negative)

    x = list(range(1, num+1, 1))
    n_labels = []
    n_values = []
    p_labels = []
    p_values = []
    for tupla in pos_terms:
        lab, val = tupla
        p_labels.append(lab)
        p_values.append(val)
    for tupla in neg_terms:
        lab, val = tupla
        n_labels.append(lab)
        n_values.append(-val)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(x, p_values, tick_label=p_labels)
    plt.title('Most common Positive')
    plt.ylabel('TfIdf Count')
    fig.savefig('Positive_terms.png')
    plt.show()

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(x, n_values, tick_label=n_labels)
    plt.ylabel('TfIdf Count')
    plt.title('Most common Negative')
    fig.savefig('Negative_terms.png')
    plt.show()


def get_sns_cfmatrix(y_test, y_pred):
    conf_mat = confusion_matrix(y_test, y_pred)
    conf_mat_df = pd.DataFrame(conf_mat, index=['pos', 'neg'], columns=['pos', 'neg'])
    conf_mat_df.index.name = 'Actual'
    conf_mat_df.columns.name = 'Predicted'
    plot = sns.heatmap(conf_mat_df, annot=True, cmap='GnBu',
                annot_kws={"size": 20}, fmt='g', cbar=False)
    plt.show()


def dump_to_file(filename, y_pred, dataset):
    with open(filename, mode="w", newline="") as csvfile:
        # Headers
        fieldnames = ['Id', 'Predicted']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for id, cls in zip(dataset.index.values.tolist(), y_pred):
            writer.writerow({'Id': str(id), 'Predicted': str(cls)})


# HERE THE STEMMER CLASS
class ItalianStemmerTokenizer(object):
    def __init__(self):
        self.stemmer = ItalianStemmer()

    def __call__(self, document):
        lemmas = []
        re_digit = re.compile("[0-9\']")
        re_no_space = re.compile('[.;:!?,\"()\[\]]')


        for t in word_tokenize(document):

            t = re_digit.sub(" ", t)
            t = re_no_space.sub("", t.lower())
            t = t.strip()  # it remove spaces before and after the characters

            if t != '':
                lemma = self.stemmer.stem(t)  # apply the stemmer
                lemmas.append(lemma)

        return lemmas


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
'''
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
'''
# 2) Vectorization
print('Vectorization:')
stem = ItalianStemmerTokenizer()
stop_words = get_stop_words('it')
#stop_words = ['milano', 'un', 'tutti', 'ci', 'lo', 'era', 'ma']
stop_words.extend(['abbi', 'abbiam', 'adess', 'allor', 'ancor', 'avemm', 'avend', 'aver', 'avess', 'avesser', 'avessim',
                   'avest', 'avet', 'avev', 'avevam', 'avra', 'avrann', 'avre', 'avrebb', 'avrebber', 'avrem', 'avremm',
                   'avrest', 'avret', 'avut', 'com', 'contr', 'dentr', 'ebber', 'eran', 'erav', 'eravam', 'essend',
                   'fac', 'facc', 'facess', 'facessim', 'facest', 'fann', 'far', 'fara', 'farann', 'farebb', 'farebber',
                   'farem', 'farest', 'fec', 'fecer', 'fin', 'foss', 'fosser', 'fossim', 'fost', 'fumm', 'fur', 'hann',
                   'lor', 'nostr', 'perc', 'poc', 'poch', 'qual', 'quant', 'quas', 'quell', 'quest', 'quind', 'sar',
                   'sara', 'sarann', 'sare', 'sarebb', 'sarebber', 'sarem', 'sarest', 'senz', 'siam', 'sian', 'siat',
                   'siet', 'son', 'sopr', 'sott', 'stand', 'stann', 'star', 'stara', 'starann', 'starebb', 'starebber',
                   'starem', 'starest', 'stav', 'stavam', 'stemm', 'stess', 'stesser', 'stessim', 'stest', 'stett',
                   'stetter', 'sti', 'stiam', 'tutt', 'vostr', 'stat', 'bagn', 'soggiorn', 'propr', 'quand', 'alberg',
                   'hotel', 'post', 'sal', 'venez', 'min', 'due', 'pot', 'cam', 'volt', 'pied', 'camer', 'vist', 'lett',
                   'reception', 'dop', 'cit', 'sol', 'ver', 'ogni', 'dir', 'pass', 'cos', 'fat', 'colazion', 'personal',
                   'stazion', 'trov'])

tfidf_vec = TfidfVectorizer(tokenizer=stem, ngram_range=(1,2), binary=True, stop_words=stop_words, min_df=8)
print('Vectorizing dev set..')
X_dev = tfidf_vec.fit_transform(dev_set['text'])
print('Vectorizing eval set..')
X_eva = tfidf_vec.transform(eva_set['text'])

# Classification

print('Start Classification:')
labels = dev_set['class']
# 1) Split data in Training and test set
X_train, X_test, y_train, y_test = train_test_split(X_dev, labels, test_size=0.2)
# 2) Define a classifier
clf = SGDClassifier()
# clf = LinearSVC()
# clf = RandomForestClassifier(n_estimators=100, min_impurity_decrease=0.12)
# clf = GradientBoostingClassifier(n_estimators=250, min_impurity_decrease=0.2)
# clf = MultinomialNB()
# Performing the Grid Search Cross Valitaion
param_grid = {'loss':['hinge','modified_huber', 'squared_hinge'], 'warm_start':[True, False], 'penalty':['l1', 'l2']}
# param_grid = {'C': [0.01, 0.1, 0.3, 0.5, 0.75, 0.8, 0.85, 0.9, 1]}
gridsearch = GridSearchCV(clf, param_grid, scoring='f1_weighted', cv=5)
clf = gridsearch.fit(X_train, y_train)
print("Best model configuration is:")
print(clf.best_params_)
print("with f1 = ", clf.best_score_)
# 3) Output the statistics & perform the cross validation
print_stats(y_test, clf.predict(X_test))
f1_cv = cross_val_score(clf, X_dev, labels, cv=3, scoring='f1_weighted')
mean_f1 = f1_cv.mean()
std_f1 = f1_cv.std()
print(f"f1 (statistics): {mean_f1:.2f} (+/- {std_f1:.2f})")
# Get seaborn confusion matrix
get_sns_cfmatrix(y_test, clf.predict(X_test))
# 4) Fit the final model with the best hyperparameters found
print('Fitting final model...')
final_model = clf.best_estimator_
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_eva)

# Showing the post common terms in both categories
get_top_terms(5, final_model, tfidf_vec)

# Dumping to file
print('Dumping in file...')
dump_to_file('result.csv', y_pred, eva_set)
print('Process done in %s seconds' % (time.time() - start_time))


