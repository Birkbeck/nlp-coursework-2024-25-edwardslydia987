import pandas as pd
import re
import string
import nltk
import contractions
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')
 

def subset_and_rename(file):
    df = pd.read_csv(file)
    df['party']  = df['party'].replace('Labour (Co-op)', 'Labour')
    df = df[df.party != 'Speaker'] 
    common_party = df['party'].value_counts().nlargest(4)
    df = df[df['party'].isin(common_party.index)]
    df = df[df.speech_class == 'Speech']
    df = df[df['speech'].str.len() < 1000]

    return df

def custom_tokenizer(text):
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    text = contractions.fix(text)
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(rf'[{re.escape(string.punctuation)}]', '', text)    
    
    tokens = word_tokenize(text)
    
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return tokens

def rf_classifier(x_train, x_test, y_train, y_test):
    rf_classifier = RandomForestClassifier(n_estimators = 300)
    rf_classifier.fit(x_train, y_train)
    rf_predictions = rf_classifier.predict(x_test)
    rf_f1_score = f1_score(y_test, rf_predictions, average = 'macro')
    rf_report = classification_report(y_test, rf_predictions)
    
    return (rf_f1_score, rf_report)

def svm_classifier(x_train, x_test, y_train, y_test):
    svm_classifier = SVC(kernel = 'linear')
    svm_classifier.fit(x_train, y_train)
    svm_predictions = svm_classifier.predict(x_test)
    svm_f1_score = f1_score(y_test, svm_predictions, average = 'macro')
    svm_report = classification_report(y_test, svm_predictions)
    
    return (svm_f1_score, svm_report)

def ngram_classifiers(ngram_x_train, ngram_x_test, ngram_y_train, ngram_y_test):
    ngram_rf_classifier = RandomForestClassifier(n_estimators = 300)
    ngram_rf_classifier.fit(ngram_x_train, ngram_y_train)
    ngram_rf_predictions = ngram_rf_classifier.predict(ngram_x_test)
    ngram_rf_f1_score = f1_score(ngram_y_test, ngram_rf_predictions, average = 'macro')
    ngram_rf_report = classification_report(ngram_y_test, ngram_rf_predictions)
    
    ngram_svm_classifier = SVC(kernel = 'linear')
    ngram_svm_classifier.fit(ngram_x_train, ngram_y_train)
    ngram_svm_predictions = ngram_svm_classifier.predict(ngram_x_test)
    ngram_svm_f1_score = f1_score(ngram_y_test, ngram_svm_predictions, average = 'macro')
    ngram_svm_report = classification_report(ngram_y_test, ngram_svm_predictions)

    return (ngram_rf_report, ngram_svm_report)


if __name__ == "__main__":
    df = subset_and_rename('hansard40000.csv')
    print(df.shape)
    
    vectorizer = TfidfVectorizer(stop_words = 'english', max_features = 3000)
    x = vectorizer.fit_transform(df['speech'])
    y = df['party']

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 26, stratify = y)
    rf_f1_score, rf_report = rf_classifier(x_train, x_test, y_train, y_test)
    svm_f1_score, svm_report = svm_classifier(x_train, x_test, y_train, y_test)
    print(rf_f1_score)
    print(rf_report)
    print(svm_f1_score)
    print(svm_report)

    ngram_vectorizer = TfidfVectorizer(stop_words = 'english', max_features = 3000, ngram_range = (1, 3))
    ngram_x = ngram_vectorizer.fit_transform(df['speech'])
    ngram_y = df['party']

    ngram_x_train, ngram_x_test, ngram_y_train, ngram_y_test = train_test_split(ngram_x, ngram_y, random_state = 26, stratify = ngram_y)
    ngram_rf_report, ngram_svm_report = ngram_classifiers(ngram_x_train, ngram_x_test, ngram_y_train, ngram_y_test)
    print(ngram_rf_report)
    print(ngram_svm_report)

    custom_vectorizer = TfidfVectorizer(tokenizer = custom_tokenizer, lowercase = False, max_features = 3000)
    custom_x = custom_vectorizer.fit_transform(df['speech'])
    custom_y = df['party']

    custom_x_train, custom_x_test, custom_y_train, custom_y_test = train_test_split(custom_x, custom_y, random_state = 26, stratify = custom_y)

    custom_rf_f1_score, custom_rf_report = rf_classifier(custom_x_train, custom_x_test, custom_y_train, custom_y_test)
    custom_svm_f1_score, custom_svm_report = svm_classifier(custom_x_train, custom_x_test, custom_y_train, custom_y_test)
    
    print(custom_rf_f1_score)
    print(custom_rf_report)
    print(custom_svm_f1_score)
    print(custom_svm_report)

    ngram_custom_vectorizer = TfidfVectorizer(tokenizer = custom_tokenizer, lowercase = False, max_features = 3000, ngram_range = (1, 3))
    ngram_custom_x = custom_vectorizer.fit_transform(df['speech'])
    ngram_custom_y = df['party']

    ngram_custom_x_train, ngram_custom_x_test, ngram_custom_y_train, ngram_custom_y_test = train_test_split(ngram_custom_x, ngram_custom_y, random_state = 26, stratify = ngram_custom_y)

    ngram_custom_rf_report, ngram_custom_svm_report = ngram_classifiers(ngram_custom_x_train, ngram_custom_x_test, ngram_custom_y_train, ngram_custom_y_test)
    
    print(ngram_custom_rf_report)
    print(ngram_custom_svm_report)