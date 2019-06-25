from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import grid_search
import data_clean
import data_load
import models
import numpy as np
import config
import pickle


# Seperate feature and label; seperate trian and test data
class DataPrep:
    def __init__(self):
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []
        self.prepare_data()

    def prepare_data(self):
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
    # Data Prep
    X_train = DataPrep().X_train
    y_train = DataPrep().y_train
    X_test = DataPrep().X_test
    y_test = DataPrep().y_test

    # Train, MultinomialNaiveBayesCV
    text_clf = models.MultinomialNaiveBayesCV().clf
    text_clf = text_clf.fit(X_train, y_train)

    # Test, Performance of NB Classifier
    predicted = text_clf.predict(X_test)
    score = np.mean(predicted == y_test)
    print(score)

    # grid search
    gs_clf = grid_search.Tune(text_clf).clf
    gs_clf = gs_clf.fit(X_train, y_train)
    print(gs_clf.best_score_)
    print(gs_clf.best_params_)

    with open(config.PICKLEFILE, 'rb') as f:
        dogDict = pickle.load(f)
    for data in dogDict:
        print(data)


'''
    # Train and Test, SGDClassifierCV
    text_clf = models.SGDClassifierCV().clf
    text_clf = text_clf.fit(X_train, y_train)
    predicted = text_clf.predict(X_test)
    score = np.mean(predicted == y_test)
    print(score)
'''

if __name__ == '__main__':
    main()
