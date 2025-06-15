import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

df = pd.read_csv('hansard40000.csv')
df['party']  = df['party'].replace('Labour (Co-op)', 'Labour')
df = df[df.party != 'Speaker']

common_party = df['party'].value_counts().nlargest(4)
df = df[df['party'].isin(common_party.index)]

df = df[df.speech_class == 'Speech']

df = df[df['speech'].str.len() < 1000]


print(df.shape)


vectorizer = TfidfVectorizer(stop_words = 'english', max_features = 3000)
x = vectorizer.fit_transform(df['speech'])
y = df['party']

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 26, stratify = y)
