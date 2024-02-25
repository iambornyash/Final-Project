from flask import Flask,request,render_template
import pandas as pd
import numpy as np
import pickle

df = pickle.load(open('df.pkl','rb'))
similarity = pickle.load(open('similarity.pkl','rb'))


def recommendation(song_df):
    idx = df[df['Title'] == song_df].index[0]
    distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])
    # print(distances)
    songs = []
    for m_id in distances[1:21]:
        songs.append(df.iloc[m_id[0]].Title)

    return songs

# flask app

app = Flask(__name__)
@app.route('/')
def index():
    songs = list(df['Title'].values)
    return render_template('index.html', names=songs)

@app.route('/recom',methods=['POST'])
def recom():
    song = request.form['names']
    songs = recommendation(song)
    print(songs)
    return render_template('index.html',songs=songs)


# python main
if __name__ == "__main__":
    app.run(debug=True)