# import string
import pandas as pd
import LemmaTokenizer
import csv
import seaborn as sns
from stop_words import get_stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score


def print_stats(y_test, y_pred):
    print('Classification results:')
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(f1_score(y_test, y_pred, average='weighted'))


def load_data(file):
    df = pd.read_csv(file, sep=',')
    return df


def into_tfidf(data ,min_df, max_df):
    lemma = LemmaTokenizer.LemmaTokenizer()
    stop_words = get_stop_words('it')
    stop_words.extend(('quantum', 'hotel'))
    vectorizer = TfidfVectorizer(tokenizer=lemma, lowercase=True, min_df=min_df, max_df=max_df,
                                 stop_words=stop_words)
    x_tfidf = vectorizer.fit_transform(data)
    return x_tfidf


def dim_reduction(tfidf):
    svd = TruncatedSVD(n_components=15, random_state=25)
    # tsne = TSNE(n_components=3)
    # pipeline = make_pipeline(svd, tsne)
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


def classify_dataset(eval_file):
    eval_df = load_data(eval_file)
    X_tfidf = into_tfidf(eval_df['text'], 0.01, 0.9)
    red_data = dim_reduction(X_tfidf)
    class_pred = clf.predict(red_data)
    dump_to_file('result.csv', class_pred, eval_df)


data = load_data('development.csv')
print(data.head())
print(data.describe())
print('Text example before preprocessing:')
print(data['text'][:3])

print('Vectorizing text...')
X_tfidf = into_tfidf(data['text'], 0.01, 0.9)
print('tfidf shape:')
print(X_tfidf.shape)
labels = data['class']

print('Dimensionality Reduction...')
# Dimesionality Reduction
red_tfidf = dim_reduction(X_tfidf)
print('tfidf shape after reduction: ', red_tfidf.shape)

# Split data in Training and test set
print('Start Classification...')
X_train, X_test, y_train, y_test = train_test_split(red_tfidf,
                                                    labels, test_size=0.20)
# Define a classifier
# clf = RandomForestClassifier(n_estimators=500, criterion='entropy', max_depth=20)
clf = GradientBoostingClassifier(n_estimators=250, min_impurity_decrease=0.12)
# clf = SVC(gamma='scale')
# Fitting classifier
clf.fit(X_train, y_train)
# Make the prediction
y_pred = clf.predict(X_test)
# Print results
print_stats(y_test, y_pred)
# Using cross validation
f1_cv = cross_val_score(clf, red_tfidf, labels, cv=5, scoring='f1_weighted')
mean_f1 = f1_cv.mean()
std_f1 = f1_cv.std()
print(f"f1 (statistics): {mean_f1:.2f} (+/- {std_f1:.2f})")
# Applying the classificator at the evaluation dataset
# and dump to file
print('Apply Classifier to Evaluation Dataset...')
classify_dataset('evaluation.csv')
print('Done!')


