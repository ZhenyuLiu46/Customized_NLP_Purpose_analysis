# from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
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
            cleaned_data_X, cleaned_data_y, test_size=0.20, random_state=41)


def main():
    # Data preperation
    logging.info('called')
    X_train = DataPrep().X_train
    y_train = DataPrep().y_train
    X_test = DataPrep().X_test
    y_test = DataPrep().y_test

    # Train, MultinomialNaiveBayesCV
    weight = np.array(compute_sample_weight(
        class_weight='balanced', y=y_train)).tolist()
    # To adjust relative weight further
    for i in range(len(y_train)):
        if y_train[i] == 1:
            weight[i] *= 1

    text_clf = models.MultinomialNaiveBayesCV().clf
    text_clf = text_clf.fit(
        X_train, y_train, **{'clf__sample_weight': weight})

    # Test, Performance of MultinomialNaiveBayesCV Classifier
    predicted = text_clf.predict(X_test)
    score = np.mean(predicted == y_test)  # Accuracy
    prec_score_bi = precision_score(
        y_test, predicted, pos_label=1, average='binary')  # precision_bi, 1 for true alerts
    recall_score_bi = recall_score(
        y_test, predicted, pos_label=1, average='binary')

    f2_score = 5 * (prec_score_bi * recall_score_bi) / \
        (4*prec_score_bi + recall_score_bi)

    logging.critical('MultinomialNaiveBayes accuracy score: %s', score)
    logging.critical(
        'MultinomialNaiveBayes precision score: %s', prec_score_bi)
    logging.critical(
        'MultinomialNaiveBayes recall score: %s', recall_score_bi)
    logging.critical(
        'MultinomialNaiveBayes f2 score: %s', f2_score)

    print('MultinomialNaiveBayes accuracy score: ', score)
    print('MultinomialNaiveBayes precision score: ', prec_score_bi)
    print('MultinomialNaiveBayes recall score: ', recall_score_bi)
    print('MultinomialNaiveBayes f2 score: ', f2_score)

    # Train, SGDClassifierCV
    print("===================")
    # To adjust relative weight further
    for i in range(len(y_train)):
        if y_train[i] == 1:
            weight[i] *= 2

    text_clf_SVM = models.SGDClassifierCV().clf
    text_clf_SVM = text_clf_SVM.fit(
        X_train, y_train, **{'clf-svm__sample_weight': weight})

    # Test, Performance of SGDClassifierCV Classifier

    predicted_SVM = text_clf_SVM.predict(X_test)
    score_SVM = np.mean(predicted_SVM == y_test)  # Accuracy
    prec_score_bi_SVM = precision_score(
        y_test, predicted_SVM, pos_label=1, average='binary')  # precision_bi, 1 for true alerts
    recall_score_bi_SVM = recall_score(
        y_test, predicted_SVM, pos_label=1, average='binary')

    f2_score_SVM = 5 * (prec_score_bi_SVM * recall_score_bi_SVM) / \
        (4*prec_score_bi_SVM + recall_score_bi_SVM)

    logging.critical('SVM_SGD accuracy score: %s', score_SVM)
    logging.critical(
        'SVM_SGD precision score: %s', prec_score_bi_SVM)
    logging.critical(
        'SVM_SGD recall score: %s', recall_score_bi_SVM)
    logging.critical(
        'SVM_SGD f2 score: %s', f2_score_SVM)

    print('SVM_SGD accuracy score: ', score_SVM)
    print('SVM_SGD precision score: ', prec_score_bi_SVM)
    print('SVM_SGD recall score: ', recall_score_bi_SVM)
    print('SVM_SGD f2 score: ', f2_score_SVM)

    '''
    # grid search: perform hyper parameter tuning
    gs_clf = grid_search.Tune(text_clf).clf
    gs_clf = gs_clf.fit(X_train, y_train)
    logging.critical('grid search best score: %s', gs_clf.best_score_)
    print(gs_clf.best_score_)
    logging.critical('grid search best params: %s', gs_clf.best_params_)
    print(gs_clf.best_params_)
    '''
    '''
    # To test picklefile output
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
