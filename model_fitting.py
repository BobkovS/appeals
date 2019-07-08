import os
import pickle as pkl
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from utils import _tokenizer

np.random.seed = 42

warnings.filterwarnings("ignore")

# Загружаем данные
data = pd.read_csv(os.path.join('data', 'fully_prepared_data.csv'))

# Шафлим датафрейм
data = data.sample(frac=1).reset_index(drop=True)

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

# Инициализируем объекты XGBClassifier
clf_category = LinearSVC()
clf_executor = LinearSVC()
clf_theme = LinearSVC()

# Учим модели первого уровня
clf_category.fit(X_train, Y_category_train)
clf_executor.fit(X_train, Y_executor_train)
clf_theme.fit(X_train, Y_theme_train)

# Определяем точность моделей первого уровня
clf_category_accuracy = clf_category.score(X_test, Y_category_test)
clf_executor_accuracy = clf_executor.score(X_test, Y_executor_test)
clf_theme_accuracy = clf_theme.score(X_test, Y_theme_test)

print('Prediction accuracy of lvl1 models: category = {0}, executor = {1}, theme = {2}\n'.format(clf_category_accuracy,
                                                                                                 clf_executor_accuracy,
                                                                                                 clf_theme_accuracy))

# Сохраняем модели первого уровня
pkl.dump(clf_category, open(os.path.join('models', 'classifiers', 'lvl1', 'clf_category.pkl'), 'wb'),
         pkl.HIGHEST_PROTOCOL)
pkl.dump(clf_executor, open(os.path.join('models', 'classifiers', 'lvl1', 'clf_executor.pkl'), 'wb'),
         pkl.HIGHEST_PROTOCOL)
pkl.dump(clf_theme, open(os.path.join('models', 'classifiers', 'lvl1', 'clf_theme.pkl'), 'wb'), pkl.HIGHEST_PROTOCOL)
