from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import grid_search
import models
import numpy as np
import config
import pickle
import sys
import logging
import traceback

# Seperate feature and label; seperate trian and test data from Picklefile


class DataPrep:
    def __init__(self):
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []
        self.prepare_data()

    def prepare_data(self):
        logging.info('called')
        with open(config.PICKLEFILE, 'rb') as f:
            cleaned_all_data = pickle.load(f)
        cleaned_data_X = []
        cleaned_data_y = []
        for data in cleaned_all_data:
            cleaned_data_X.append(data[0])
            cleaned_data_y.append(data[1])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            cleaned_data_X, cleaned_data_y, test_size=0.33, random_state=42)


def main():
    # Data preperation
    logging.info('called')
    X_train = DataPrep().X_train
    y_train = DataPrep().y_train
    X_test = DataPrep().X_test
    y_test = DataPrep().y_test

    # Train, MultinomialNaiveBayesCV
    text_clf = models.MultinomialNaiveBayesCV().clf
    text_clf = text_clf.fit(X_train, y_train)

    # Test, Performance of MultinomialNaiveBayesCV Classifier
    predicted = text_clf.predict(X_test)
    score = np.mean(predicted == y_test)
    logging.critical('MultinomialNaiveBayes score: %s', score)
    print(score)

    # grid search: perform hyper parameter tuning
    gs_clf = grid_search.Tune(text_clf).clf
    gs_clf = gs_clf.fit(X_train, y_train)
    logging.critical('grid search best score: %s', gs_clf.best_score_)
    print(gs_clf.best_score_)
    logging.critical('grid search best params: %s', gs_clf.best_params_)
    print(gs_clf.best_params_)

    '''
    #To test picklefile output
    with open(config.PICKLEFILE, 'rb') as f:
        allData = pickle.load(f)
    for data in allData:
        print(data)
    '''


'''
    # Train and Test, SGDClassifierCV
    text_clf = models.SGDClassifierCV().clf
    text_clf = text_clf.fit(X_train, y_train)
    predicted = text_clf.predict(X_test)
    score = np.mean(predicted == y_test)
    print(score)
'''

if __name__ == '__main__':
    # set logging config
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)s] %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename='info.log', filemode='a',
                        level=logging.DEBUG, format=FORMAT)
    try:
        main()
    except Exception as e:
        traceback.print_exc()  # console print exception
        logging.exception("Exception occurred")  # log exception
