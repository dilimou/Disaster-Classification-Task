# Disaster Classification Task

## Description:
The aim of this project is to build a Natural Language Processing tool that categorize disaster response messages.

The project is divided into multiple parts:
Data Processing, ETL Pipeline to extract data from source, clean data and save them in a proper data structure.
Machine Learning Pipeline to train a model which is able to classify text messages in 36 categories.
Web Application using Flask to show model results and predictions in real time.

Data:
The data in this project comes from Figure Eight.

## Folder Structure:

- app
    - templates
        - master.html # main page of application
        - go.html # classification result page of application
    - run.py # Flask file that runs application
data- 
    - disaster_categories.csv # data to process
    - disaster_messages.csv # data to process
    - process_data.py
    - Disaster_Response.db # database with the clean data
- models
    - train_classifier.py
    - classifier.pkl # saved model
- README.md

## Requirements:

- Pandas
- Matplotlib
- Json
- Plotly
- Nltk
- Flask
- Sklearn
- Sqlalchemy
- Sys
- Re
- Pickle

## Instructions:

In a terminal, go to the 'app' folder, run the following command:
- python run.py

Go to [http://0.0.0.0:3001/](http://0.0.0.0:3001/)

Run the following commands in the project's root directory to recreate the database and model:
- python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
- python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl


