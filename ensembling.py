import os
import pickle as pkl
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier, RidgeClassifier, SGDClassifier
from sklearn.svm import LinearSVC

from utils import _tokenizer

np.random.seed = 42

warnings.filterwarnings("ignore")

# Загружаем данные
data = pd.read_csv(os.path.join('data', 'fully_prepared_data.csv'))

# Шафлим датафрейм
data = data.sample(frac=1).reset_index(drop=True)

for i in range(10):

    # Разбиваем датафрейм на трейн и тест с отношением 0.9/0.1
    train_index = np.random.rand(len(data)) < 0.9
    train_data = data[train_index].reset_index(drop=True)
    test_data = data[~train_index].reset_index(drop=True)

    test_data.to_csv(os.path.join('data', 'test_data.csv'), index=False)

    # Инициализируем и обучаем векторайзер
    vectorizer = TfidfVectorizer(tokenizer=_tokenizer)
    vectorizer.fit(data.text)
    pkl.dump(vectorizer, open(os.path.join('models', 'vectorizer.pkl'), 'wb'), pkl.HIGHEST_PROTOCOL)

    # Инициализируем X и Y
    X_train, X_test = vectorizer.transform(train_data.text), vectorizer.transform(test_data.text)
    X_train, X_test = pd.DataFrame(X_train.toarray()), pd.DataFrame(X_test.toarray())
    Y_category_train, Y_category_test = train_data.category, test_data.category
    Y_executor_train, Y_executor_test = train_data.executor, test_data.executor
    Y_theme_train, Y_theme_test = train_data.theme, test_data.theme

    # Инициализируем Модели
    svc = LinearSVC()
    pac = PassiveAggressiveClassifier()
    ridge = RidgeClassifier()
    sgdc = SGDClassifier()
    etc = ExtraTreesClassifier()

    svc.fit(X_train, Y_category_train)
    pac.fit(X_train, Y_category_train)
    ridge.fit(X_train, Y_category_train)
    sgdc.fit(X_train, Y_category_train)
    etc.fit(X_train, Y_category_train)

    predictions, result_predictions = [], []
    svc_predict = svc.predict(X_test)
    pac_predict = pac.predict(X_test)
    ridge_predict = ridge.predict(X_test)
    sgdc_predict = sgdc.predict(X_test)
    etc_predict = etc.predict(X_test)

    for i in range(len(svc_predict)):
        predictions.append(svc_predict[i])
        predictions.append(pac_predict[i])
        predictions.append(ridge_predict[i])
        predictions.append(sgdc_predict[i])
        predictions.append(etc_predict[i])
        result_predictions.append(Counter(predictions).most_common(1)[0][0])
        predictions = []

    correct_answers = 0
    for i in range(len(result_predictions)):
        if result_predictions[i] == Y_category_test[i]:
            correct_answers += 1

    print('svc = {0}'.format(svc.score(X_test, Y_category_test)))
    print('pac = {0}'.format(pac.score(X_test, Y_category_test)))
    print('ridge = {0}'.format(ridge.score(X_test, Y_category_test)))
    print('sgdc = {0}'.format(sgdc.score(X_test, Y_category_test)))
    print('etc = {0}'.format(etc.score(X_test, Y_category_test)))

    print("Result accuracy = {0} \n\n".format(correct_answers / len(result_predictions)))
