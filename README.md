# Disaster Response Pipeline Project
Disaster Response Pipeline Project for Udacity Data Science Nanodegree

### Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Instructions](#instructions)

## Installation <a name="installation"></a>

The following nltk packages need to be installed for this project:
* punkt
* wordnet
* stopwords


## Project Motivation<a name="motivation"></a>

This project applies a number of data science techniques, including NLP, machine learning and ETL processing to analyse messages sent by people during natural disasters.  This results in a model that can classify messages be genre.  In turn, this information is available via an API.  Analysis of the model output may be useful to disaster relief agencies.

## File Descriptions <a name="files"></a>

Files are available via the following folders:
1. data
    - disaster_categories.csv: dataset of all categories 
    - disaster_messages.csv: dataset of all messages
    - process_data.py: ETL pipeline load, cleanse, and save data to SQLite database
    - DisasterResponse.db: SQLite database output of the ETL pipeline containing messages and categories data
2. models
    - train_classifier.py: machine learning pipeline scripts to train and export a classifier
3. app
    - run.py: flask file for web application
    - templates: html files for web application

## Results<a name="results"></a>

1. An ETL pipleline was built to read data from two csv files, clean data, and save data into a SQLite database.
2. The machine learning random forest classification pipepline was developed to build a model that performs multi-output classification on the dataset categories.
3. The Flask app applies data visualizations and classifies a message entered by a user via the web page.  Visualisations include both genre and category counts.

## Instructions:<a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
