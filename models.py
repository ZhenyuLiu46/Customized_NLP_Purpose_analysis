from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
# for developing test
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import SGDClassifier
import numpy as np
import data_clean  # just for now testing
import data_load


class MultinomialNaiveBayesCV():
    def __init__(self):
        self.clf = Pipeline([('vect', CountVectorizer()),
                             ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])


class SGDClassifierCV():
    def __init__(self):
        self.clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                             ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=5, random_state=42))])


'''
# test
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

text_clf = MultinomialNaiveBayes().clf
text_clf = text_clf.fit(twenty_train.data, twenty_train.target)

# Performance of NB Classifier
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
predicted = text_clf.predict(twenty_test.data)
score = np.mean(predicted == twenty_test.target)
print(score)
'''
