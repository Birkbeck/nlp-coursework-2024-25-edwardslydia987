import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report

def subset_and_rename(file):
    df = pd.read_csv(file)
    df['party']  = df['party'].replace('Labour (Co-op)', 'Labour')
    df = df[df.party != 'Speaker']
    
    common_party = df['party'].value_counts().nlargest(4)
    df = df[df['party'].isin(common_party.index)]

    df = df[df.speech_class == 'Speech']

    df = df[df['speech'].str.len() < 1000]

    return df



if __name__ == "__main__":
    df = subset_and_rename('hansard40000.csv')
    print(df.shape)
    
    vectorizer = TfidfVectorizer(stop_words = 'english', max_features = 3000)
    x = vectorizer.fit_transform(df['speech'])
    y = df['party']

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 26, stratify = y)

    rf_classifier = RandomForestClassifier(n_estimators = 300)
    rf_classifier.fit(x_train, y_train)
    rf_predictions = rf_classifier.predict(x_test)
    rf_f1_score = f1_score(y_test, rf_predictions, average = 'macro')
    rf_report = classification_report(y_test, rf_predictions)
    print(rf_f1_score)
    print(rf_report)

    svm_classifier = SVC(kernel = 'linear')
    svm_classifier.fit(x_train, y_train)
    svm_predictions = svm_classifier.predict(x_test)
    svm_f1_score = f1_score(y_test, svm_predictions, average = 'macro')
    svm_report = classification_report(y_test, svm_predictions)
    print(svm_f1_score)
    print(svm_report)

    ngram_vectorizer = TfidfVectorizer(stop_words = 'english', max_features = 3000, ngram_range = (1, 3))
    ngram_x = ngram_vectorizer.fit_transform(df['speech'])
    ngram_y = df['party']

    ngram_x_train, ngram_x_test, ngram_y_train, ngram_y_test = train_test_split(ngram_x, ngram_y, random_state = 26, stratify = y)

    ngram_rf_classifier = RandomForestClassifier(n_estimators = 300)
    ngram_rf_classifier.fit(ngram_x_train, ngram_y_train)
    ngram_rf_predictions = rf_classifier.predict(ngram_x_test)
    ngram_rf_f1_score = f1_score(ngram_y_test, ngram_rf_predictions, average = 'macro')
    ngram_rf_report = classification_report(ngram_y_test, ngram_rf_predictions)
    print(ngram_rf_report)

    ngram_svm_classifier = SVC(kernel = 'linear')
    ngram_svm_classifier.fit(ngram_x_train, ngram_y_train)
    ngram_svm_predictions = svm_classifier.predict(ngram_x_test)
    ngram_svm_f1_score = f1_score(ngram_y_test, ngram_svm_predictions, average = 'macro')
    ngram_svm_report = classification_report(ngram_y_test, ngram_svm_predictions)
    print(ngram_svm_report)