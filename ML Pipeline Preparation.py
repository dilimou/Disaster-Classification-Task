#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# In[22]:


# import libraries
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


# In[4]:


# load data from database
engine = create_engine('sqlite:///DisasterClassification.db')
df = pd.read_sql_table("DisaserCategory", engine)
X = df["message"]
Y = df.iloc[:,4:]
df.head()
X.head()
Y.head()


# ### 2. Write a tokenization function to process your text data

# In[5]:


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


# In[6]:


print(X[4])
print(tokenize(X[4]))


# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

# In[7]:


pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])


# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# In[8]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
pipeline.fit(X_train, Y_train)


# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# In[9]:


Y_pred = pipeline.predict(X_test)


# In[10]:


for ix, col in enumerate(Y.columns):
    print(col)
    print(classification_report(Y_test[col], Y_pred[:,ix]))

avg = (Y_pred == Y_test).mean().mean()
print("Accuracy Overall:\n", avg)


# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[11]:


pipeline.get_params()


# In[17]:


parameters = {
        'vect__max_df':[0.75,1.0],
        'clf__estimator__n_estimators': [20, 50]
    }

cv = GridSearchCV(pipeline, param_grid=parameters)


# In[18]:


# Train model
cv.fit(X_train, Y_train)


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# In[19]:


Y_pred_new = cv.predict(X_test)


# In[20]:


for ix, col in enumerate(Y.columns):
    print(col)
    print(classification_report(Y_test[col], Y_pred_new[:,ix]))

avg = (Y_pred_new == Y_test).mean().mean()
print("Accuracy Overall:\n", avg)


# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# In[ ]:





# ### 9. Export your model as a pickle file

# In[23]:


pickle.dump(cv, open("DisasterMessageClassifier.pkl", 'wb'))


# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

# In[ ]:




