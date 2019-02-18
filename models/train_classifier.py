# import libraries
import sys
import pandas as pd
import numpy as np
import nltk
import re
from nltk.tokenize import word_tokenize, WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle

nltk.download(['punkt', 'wordnet','stopwords'])

def load_data(database_filepath):
    '''
    Load data from database as dataframe
    Input:
        database_filepath: Location of sql database
    Output:
        X: Message data (features)
        Y: Categories (target)
        category_names: Category labels
    '''
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterMessages', engine)
    #df.drop('original', axis=1, inplace=True) # A lot of NaN values for this column do dropping it
    #df.dropna(how='any', inplace=True)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns
    
    return X, Y, category_names

def tokenize(text):
    '''
    input:
        text: Text data for tokenization.
    output:
        tokenized_clean: List of results after tokenization.
    '''
    # Normalize Text
    text = re.sub(r"[^a-zA-Z0-9]", ' ', text.lower())
    
    # Tokenize
    words = word_tokenize(text)
    
    # Remove Stopwords
    stopwordlist = list(set(stopwords.words('english')))
    words = [x for x in words if x not in stopwordlist]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(x, pos='n').strip() for x in words]
    tokenized_clean = [lemmatizer.lemmatize(x, pos='v').strip() for x in lemmatized]
    
    return tokenized_clean

def build_model():
    '''
    Run Machine Learning pipeline using ifidf, random forest, and gridsearch
    Input: None
    Output:
        cv: GridSearchCV output
    '''
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                       ('tfidf', TfidfTransformer()),
                       ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    
    parameters = {'clf__estimator__min_samples_split': [2, 4],
              #'clf__estimator__max_features': ['log2', 'auto', 'sqrt', None],
              #'clf__estimator__criterion': ['gini', 'entropy'],
              #'clf__estimator__max_depth': [None, 25, 50, 100, 150, 200],
             }

    cv = GridSearchCV(estimator=pipeline, param_grid=parameters)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate model performance using test data
    Input: 
        model: Model to be evaluated
        X_test: Test data
        Y_test: Labels for Test data
        category_names: Labels for categories
    Output:
        Report of accuracy and classfication for each category
    '''
    Y_pred = model.predict(X_test)
    
    for x in range(len(category_names)):
        print("Category:", category_names[x],"\n", classification_report(Y_test.iloc[:, x].values, Y_pred[:, x]))
        print('Accuracy of %25s: %.2f' %(category_names[x], accuracy_score(Y_test.iloc[:, x].values, Y_pred[:,x])))

def save_model(model, model_filepath):
    '''
    Save model as a pickle file 
    Input: 
        model: Model to save as pickle file
        model_filepath: path of the output pickle file
    Output:
        Pickle file of model
    '''
    pickle.dump(model, open(model_filepath, "wb"))

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
