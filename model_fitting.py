import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from xgboost import XGBClassifier

from utils import _tokenizer

# Загружаем данные
data = pd.read_csv(os.path.join('data', 'fully_prepared_data.csv'))

# Шафлим датафрейм
data = data.sample(frac=1).reset_index(drop=True)

# Инициализируем и обучаем векторайзер
vectorizer = CountVectorizer(tokenizer=_tokenizer)
vectorizer.fit(data.text)

# Разбиваем датафрейм на трейн и тест с отношением 0.9/0.1
train_index = np.random.rand(len(data)) < 0.9
train_data = data[train_index]
test_data = data[~train_index]

# Инициализируем X и Y
X_train, X_test = vectorizer.transform(train_data.text), vectorizer.transform(test_data.text)
Y_category_train, Y_category_test = train_data.category, test_data.category
Y_executor_train, Y_executor_test = train_data.executor, test_data.executor
Y_theme_train, Y_theme_test = train_data.theme, test_data.theme

# Инициализируем объекты XGBClassifier
clf_category, clf_executor, clf_theme = XGBClassifier(), XGBClassifier(), XGBClassifier()

# Учим модели первого уровня
clf_category.fit(X_train, Y_category_train)
clf_executor.fit(X_train, Y_executor_train)
clf_theme.fit(X_train, Y_theme_train)

# Определяем точность моделей первого уровня
clf_category_accuracy = clf_category.score(X_test, Y_category_test)
clf_executor_accuracy = clf_executor.score(X_test, Y_executor_test)
clf_theme_accuracy = clf_theme.score(X_test, Y_theme_test)

print('Prediction accuracy: category = {0}, executor = {1}, theme = {2}'.format(clf_category_accuracy,
                                                                                clf_executor_accuracy,
                                                                                clf_theme_accuracy))

# last try: Prediction accuracy: category = 0.696969696969697, executor = 0.5393939393939394, theme = 0.5333333333333333

# TODO По тексту определяем категорию, исполнителя, тему. После чего закидываем это в новые модели,
#  где, к примеру определенная категория будет являться уже одним из параметров и так до конца
