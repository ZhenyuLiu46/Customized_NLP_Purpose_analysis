from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
# for developing test
from sklearn.datasets import fetch_20newsgroups
import numpy as np


class MultinomialNaiveBayes():
    def __init__(self):
        self.clf = Pipeline([('vect', CountVectorizer()),
                             ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])


# next step here split our data into train and test data!!!!
# test
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

text_clf = MultinomialNaiveBayes().clf
text_clf = text_clf.fit(twenty_train.data, twenty_train.target)

# Performance of NB Classifier
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
predicted = text_clf.predict(twenty_test.data)
score = np.mean(predicted == twenty_test.target)
print(score)
