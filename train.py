from sklearn.model_selection import train_test_split
import data_clean
import data_load
import models
import numpy as np


class Data:
    def __init__(self):
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []
        self.prepare()

    def prepare(self):
        input_data = data_load.PurposeDataset().all_data  # load data
        #cleaned_data_all = data_clean.CleanedPurposeDataset(input_data).all_data
        cleaned_data_X = data_clean.CleanedPurposeDataset(input_data).X
        cleaned_data_y = data_clean.CleanedPurposeDataset(input_data).y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            cleaned_data_X, cleaned_data_y, test_size=0.33, random_state=42)


def main():
    # Train, MultinomialNaiveBayesCV
    X_train = Data().X_train
    y_train = Data().y_train
    text_clf = models.MultinomialNaiveBayesCV().clf
    text_clf = text_clf.fit(X_train, y_train)

    # Test, Performance of NB Classifier
    X_test = Data().X_test
    y_test = Data().y_test
    predicted = text_clf.predict(X_test)
    score = np.mean(predicted == y_test)
    print(score)

    # Train and Test, SGDClassifierCV
    text_clf = models.SGDClassifierCV().clf
    text_clf = text_clf.fit(X_train, y_train)
    predicted = text_clf.predict(X_test)
    score = np.mean(predicted == y_test)
    print(score)


if __name__ == '__main__':
    main()
