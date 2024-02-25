import pandas as pd
import numpy as np

df = pd.read_csv('lyrics.csv')
df.head(3)

df.shape
df.describe()
df.info()
df["Title"] = df["Title"].str.title()
df.Title


df["Film"] = df['Film'].str.title()
df["Singer"] = df['Singer'].str.title()
df["Composer"] = df['Composer'].str.title()
df["Lyricist"] = df['Lyricist'].str.title()
df["Lyrics"] = df['Lyrics'].str.title()

df['Title'].fillna("Not Available", inplace=True)
df["Film"].fillna("Not Available", inplace = True)
df["Singer"].fillna("Not Available", inplace = True)
df["Composer"].fillna("Not Available", inplace = True)
df["Lyricist"].fillna("Not Available", inplace = True)
df["Lyrics"].fillna("Not Available", inplace = True)

df.isnull().sum()

chars = 20
df['Title'] = df['Title'].apply(lambda x: ' '.join(x.split()[:3] [:chars]))


df['Lyrics'] = df['Lyrics'].str.replace(r'La \t', '')
df['Lyrics'] = df['Lyrics'].str.replace(r'A \t', '')



import nltk
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


def tokenization(txt):
    tokens = nltk.word_tokenize(txt)
    stemming = [stemmer.stem(w) for w in tokens]
    return " ".join(stemming)

df['Lyrics'] = df['Lyrics'].apply(lambda x: tokenization(x))



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
tfidvector = TfidfVectorizer(analyzer='word', stop_words='english')
matrix = tfidvector.fit_transform(df['Lyrics'])
similarity = cosine_similarity(matrix)

similarity[0]


def recommendation(song_df):
    idx = df[df['Title'] == song_df].index[0]
    distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])

    songs = []
    for m_id in distances[1:21]:
        songs.append(df.iloc[m_id[0]].Title)

    return songs



import pickle

pickle.dump(similarity, open('similarity.pkl', 'wb'))
pickle.dump(df, open('df.pkl', 'wb'))