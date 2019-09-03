import collections
import copy
from itertools import product, chain

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sortedcontainers import SortedList


class ParamSearch:
    def __init__(self, p_dict):
        self.p_dict = {}

        for a, b in p_dict.items():
            if isinstance(b, collections.Sequence) and not isinstance(b, str):
                self.p_dict[a] = b
            else:
                self.p_dict[a] = [b]

        self.results = SortedList()

    def grid_search(self, keys=None):

        if keys is None:
            key_list = self.p_dict.keys()
        else:
            key_list = keys

        list_of_lists = []
        for key in key_list: list_of_lists.append([(key, i) for i in self.p_dict[key]])
        for p in product(*list_of_lists):

            if len(self.results) > 0:
                template = self.results[-1][1]
            else:
                template = {a: b[0] for a, b in self.p_dict.items()}

            if self.equal_dict(dict(p), template): continue

            yield self.overwrite_dict(dict(p), template)

    def equal_dict(self, a, b):
        for key in a.keys():
            if a[key] != b[key]: return False
        return True

    def overwrite_dict(self, new, old):
        old = copy.deepcopy(old)
        for key in new.keys(): old[key] = new[key]
        return old

    def register_result(self, result, params):
        self.results.add((result + np.random.randn() * 1e-10, params))

    def best_score(self):
        return self.results[-1][0]

    def best_param(self):
        return self.results[-1][1]


class ModelTuner:
    def crossvaltest_tf_idf(self, params, x, y_cat, y_ex, y_th):
        skf = StratifiedKFold(n_splits=5)
        score = []
        for train_index, test_index in skf.split(x, y_cat):
            x_train, x_test = x.iloc[train_index], x.iloc[test_index]
            y_cat_train, y_cat_test = y_cat.iloc[train_index], y_cat.iloc[test_index]
            y_ex_train, y_ex_test = y_ex.iloc[train_index], y_ex.iloc[test_index]
            y_th_train, y_th_test = y_th.iloc[train_index], y_th.iloc[test_index]

            vectorizer = TfidfVectorizer(**params)
            vectorizer.fit(x_train, x_test)
            x_train = vectorizer.transform(x_train)
            x_test = vectorizer.transform(x_test)

            cat_clf = LinearSVC()
            ex_clf = LinearSVC()
            th_clf = LinearSVC()

            cat_clf.fit(x_train, y_cat_train)
            ex_clf.fit(x_train, y_ex_train)
            th_clf.fit(x_train, y_th_train)

            score.append(cat_clf.score(x_test, y_cat_test))
            score.append(ex_clf.score(x_test, y_ex_test))
            score.append(th_clf.score(x_test, y_th_test))

        return np.mean(score)

    def tf_idf_vectorizer_param_tune(self, params, x, y_cat, y_ex, y_th):
        ps = ParamSearch(params)
        for param in chain(ps.grid_search(['max_df', 'min_df']),
                           ps.grid_search(['sublinear_tf', 'ngram_range'])):
            res = self.crossvaltest_tf_idf(param, x, y_cat, y_ex, y_th)
            ps.register_result(res, param)
            print(res, param, 'best:', ps.best_score(), ps.best_param())
        return ps.best_param(), ps.best_score()
