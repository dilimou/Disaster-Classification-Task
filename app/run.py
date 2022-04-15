import numpy as np
import re
import pandas as pd
import json
import nltk
nltk.download(['punkt', 'wordnet','stopwords'])
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
import plotly
from plotly.graph_objs import Bar,Marker
# from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    """
        Tokenization function. To clean the text data and remove properties not useul for analysis.
        input: raw text
        process: remove stop words, ponctuations, reduce the words to their root etc...
        Returns: clean and tokenized text
                                        """
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    
    for clean_tok in tokens:
       
        # Remove stop words
        if clean_tok in stopwords.words("english"):
            continue
            
        # Reduce words to their stems
        clean_tok = PorterStemmer().stem(clean_tok)
        
        # Reduce words to their root form
        clean_tok = lemmatizer.lemmatize(clean_tok).lower().strip()

        clean_tokens.append(clean_tok)
        
    clean_tokens = [clean_tok for clean_tok in clean_tokens if clean_tok.isalpha()]
    
    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponseDatabase', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    df_categories = df.iloc[:,4:]
    genre_cat = df.groupby('genre').count().reset_index().drop(['id','message','original'], axis=1).set_index('genre')
    genre_cat = genre_cat.T.reset_index()
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=df_categories.columns,
                    y=df_categories.count()
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=genre_cat['index'],
                    y=genre_cat['direct'],
                    name = 'direct'
                ),
                Bar(
                    x=genre_cat['index'],
                    y=genre_cat['news'],
                    name = 'news'
                ),
                Bar(
                    x=genre_cat['index'],
                    y=genre_cat['social'],
                    name = 'social'
                ),
            ],

            'layout': {
                'title': 'Distribution of Message Categories by genre',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        }
    ]
     
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()