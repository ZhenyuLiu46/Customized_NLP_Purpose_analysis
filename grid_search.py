from sklearn.model_selection import GridSearchCV
import logging


class Tune:
    def __init__(self, _clf):
        self.clf = self.search(_clf)

    def search(self, _clf):
        # parameters need to adjusted for different _clf
        parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (
            True, False), 'clf__alpha': (1e-2, 1e-3)}
        gs_clf = GridSearchCV(_clf, parameters, n_jobs=-1)
        #gs_clf = gs_clf.fit(X_train, y_train)
        logging.info('called')
        return gs_clf


'''
    # Train using best params as showed above

    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer(
        use_idf=False)), ('clf', MultinomialNB(alpha=0.01))])
    text_clf = text_clf.fit(twenty_train.data, twenty_train.target)

'''
