import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

import re
import nltk
nltk.download(['punkt', 'wordnet','stopwords'])
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])

from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, hamming_loss


def load_data(database_filepath):
    ''' load data from sql database and return feature dataframe, label-data DataFrame, labels as list'''
    
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql('SELECT * FROM DisasterResponseDatabase', con=engine)
    X = df.message.values 
    Y = df.iloc[:,4:]


def tokenize(text):
    '''tokenize input messages'''
    
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



def build_model():
    '''build pipeland, set parameter, do Gridsearch and return model'''

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # small set of parameters because of time
    parameters = {
        'vect__max_df':[0.5,0.75,1.0],
        'clf__estimator__min_samples_split': [2, 3, 4]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=10)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''evaluate model by sklearn classification_report'''
    
    Y_pred = model.predict(X_test)
    
    for ix, col in enumerate(category_names):
        print(col)
        print(classification_report(Y_test[col], Y_pred[:,ix]))

    
    acc = (Y_pred == Y_test).mean().mean()
    print("Accuracy Overall:")
    print(acc)



def save_model(model, model_filepath):
    '''save model as pickle file'''
    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()