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
    - process_data.py # Data processing script
    - DisasterResponse.db # database return after processing of the data
- models
    - train_classifier.py # Script to generate the model
    - classifier.pkl # Model generated
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

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`


2. Run the following command in the app's directory to run your web app.
    `python run.py`

Go to [http://0.0.0.0:3001/](http://0.0.0.0:3001/)

If you can access the app throuugh the(http://0.0.0.0:3001/) check your terminal. The `python run.py` result will contain the path were you can locally run the app


## Results

The home page of the app
![image](https://user-images.githubusercontent.com/23463800/163554067-1289c749-8be2-47b0-b551-faee4deaf53f.png)

![image](https://user-images.githubusercontent.com/23463800/163554100-ca5cdb52-2d98-4828-801b-ece4bd8d45ff.png)

![image](https://user-images.githubusercontent.com/23463800/163554180-a1b8ec40-f108-4291-924d-aae92242444b.png)

With the available data, the model reached an accuracy of 94%.

classification example
![image](https://user-images.githubusercontent.com/23463800/163554474-574c895e-99f4-4c45-a5f6-bb655997e47d.png)



## Licensing, Authors, Acknowledgements

All credit for the data foes to Figure Eight. For collecting and labelling all the data. You can find all necessary information on the dataset on [here](https://www.figure-eight.com/).
