import os
import pickle as pkl
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from utils import _tokenizer

np.random.seed = 42

warnings.filterwarnings("ignore")

# Инициализируем хранилище для сохранения уверенности моделей
model_accuracy = {}

# Загружаем данные
data = pd.read_csv(os.path.join('data', 'fully_prepared_data.csv'))

# Шафлим датафрейм
data = data.sample(frac=1).reset_index(drop=True)

# Отдельно запоминаем мешки слов для столбцов [Category, Executor, Theme]
category_dummies = pd.get_dummies(data.category, prefix='category')
pkl.dump(category_dummies.columns, open(os.path.join('models', 'dummies_columns', 'category_columns.pkl'), 'wb'),
         pkl.HIGHEST_PROTOCOL)

executor_dummies = pd.get_dummies(data.executor, prefix='executor')
pkl.dump(executor_dummies.columns, open(os.path.join('models', 'dummies_columns', 'executor_columns.pkl'), 'wb'),
         pkl.HIGHEST_PROTOCOL)

theme_dummies = pd.get_dummies(data.theme, prefix='theme')
pkl.dump(theme_dummies.columns, open(os.path.join('models', 'dummies_columns', 'theme_columns.pkl'), 'wb'),
         pkl.HIGHEST_PROTOCOL)

# Разбиваем датафрейм на трейн и тест с отношением 0.9/0.1
train_index = np.random.rand(len(data)) < 0.9
train_data = data[train_index].reset_index(drop=True)
test_data = data[~train_index].reset_index(drop=True)

test_data.to_csv(os.path.join('data', 'test_data.csv'), index=False)

category_dummies_train = category_dummies[train_index].reset_index(drop=True)
category_dummies_test = category_dummies[~train_index].reset_index(drop=True)

executor_dummies_train = executor_dummies[train_index].reset_index(drop=True)
executor_dummies_test = executor_dummies[~train_index].reset_index(drop=True)

theme_dummies_train = theme_dummies[train_index].reset_index(drop=True)
theme_dummies_test = theme_dummies[~train_index].reset_index(drop=True)

# Инициализируем и обучаем векторайзер
vectorizer = CountVectorizer(tokenizer=_tokenizer)
vectorizer.fit(data.text)
pkl.dump(vectorizer, open(os.path.join('models', 'vectorizer.pkl'), 'wb'), pkl.HIGHEST_PROTOCOL)

# Инициализируем X и Y
X_train, X_test = vectorizer.transform(train_data.text), vectorizer.transform(test_data.text)
X_train, X_test = pd.DataFrame(X_train.toarray()), pd.DataFrame(X_test.toarray())
Y_category_train, Y_category_test = train_data.category, test_data.category
Y_executor_train, Y_executor_test = train_data.executor, test_data.executor
Y_theme_train, Y_theme_test = train_data.theme, test_data.theme

# Инициализируем объекты XGBClassifier
clf_category, clf_executor, clf_theme = LogisticRegression(), LogisticRegression(), LogisticRegression()

# Учим модели первого уровня
clf_category.fit(X_train, Y_category_train)
clf_executor.fit(X_train, Y_executor_train)
clf_theme.fit(X_train, Y_theme_train)

# Определяем точность моделей первого уровня
clf_category_accuracy = clf_category.score(X_test, Y_category_test)
clf_executor_accuracy = clf_executor.score(X_test, Y_executor_test)
clf_theme_accuracy = clf_theme.score(X_test, Y_theme_test)

model_accuracy.update({'clf_category': clf_category_accuracy})
model_accuracy.update({'clf_executor': clf_executor_accuracy})
model_accuracy.update({'clf_theme': clf_theme_accuracy})

print('Prediction accuracy of lvl1 models: category = {0}, executor = {1}, theme = {2}\n'.format(clf_category_accuracy,
                                                                                                 clf_executor_accuracy,
                                                                                                 clf_theme_accuracy))

# Сохраняем модели первого уровня
pkl.dump(clf_category, open(os.path.join('models', 'classifiers', 'lvl1', 'clf_category.pkl'), 'wb'),
         pkl.HIGHEST_PROTOCOL)
pkl.dump(clf_executor, open(os.path.join('models', 'classifiers', 'lvl1', 'clf_executor.pkl'), 'wb'),
         pkl.HIGHEST_PROTOCOL)
pkl.dump(clf_theme, open(os.path.join('models', 'classifiers', 'lvl1', 'clf_theme.pkl'), 'wb'), pkl.HIGHEST_PROTOCOL)

# Данные для моделей второго уровня
X_train_category, X_test_category = pd.concat([X_train, category_dummies_train], axis=1), pd.concat(
    [X_test, category_dummies_test], axis=1)

X_train_executor, X_test_executor = pd.concat([X_train, executor_dummies_train], axis=1), pd.concat(
    [X_test, executor_dummies_test], axis=1)

X_train_theme, X_test_theme = pd.concat([X_train, theme_dummies_train], axis=1), pd.concat(
    [X_test, theme_dummies_test], axis=1)

# Инициализируем модели второго уровня
clf_category_executor, clf_category_theme = LogisticRegression(), LogisticRegression()

clf_executor_category, clf_executor_theme = LogisticRegression(), LogisticRegression()

clf_theme_category, clf_theme_executor = LogisticRegression(), LogisticRegression()

# Обучаем модели второго уровня
clf_category_executor.fit(X_train_category, Y_executor_train)
clf_category_theme.fit(X_train_category, Y_theme_train)

clf_executor_category.fit(X_train_executor, Y_category_train)
clf_executor_theme.fit(X_train_executor, Y_theme_train)

clf_theme_category.fit(X_train_theme, Y_category_train)
clf_theme_executor.fit(X_train_theme, Y_executor_train)

# Определяем точность моделей второго уровня
clf_category_executor_accuracy = clf_category_executor.score(X_test_category, Y_executor_test)
clf_category_theme_accuracy = clf_category_theme.score(X_test_category, Y_theme_test)

clf_executor_category_accuracy = clf_executor_category.score(X_test_executor, Y_category_test)
clf_executor_theme_accuracy = clf_executor_theme.score(X_test_executor, Y_theme_test)

clf_theme_category_accuracy = clf_theme_category.score(X_test_theme, Y_category_test)
clf_theme_executor_accuracy = clf_theme_executor.score(X_test_theme, Y_executor_test)

model_accuracy.update({'clf_category_executor': clf_category_executor_accuracy})
model_accuracy.update({'clf_category_theme': clf_category_theme_accuracy})
model_accuracy.update({'clf_executor_category': clf_executor_category_accuracy})
model_accuracy.update({'clf_executor_theme': clf_executor_theme_accuracy})
model_accuracy.update({'clf_theme_category': clf_theme_category_accuracy})
model_accuracy.update({'clf_theme_executor': clf_theme_executor_accuracy})

print('Prediction accuracy of lvl2 models: \n'
      'category_executor = {0},  category_theme = {1} \n'
      'executor_category = {2},  executor_theme = {3} \n'
      'theme_category = {4},  theme_executor = {5} \n'.format(clf_category_executor_accuracy,
                                                              clf_category_theme_accuracy,
                                                              clf_executor_category_accuracy,
                                                              clf_executor_theme_accuracy, clf_theme_category_accuracy,
                                                              clf_theme_executor_accuracy))

# Сохраняем модели второго уровня
pkl.dump(clf_category_executor, open(os.path.join('models', 'classifiers', 'lvl2', 'clf_category_executor.pkl'), 'wb'),
         pkl.HIGHEST_PROTOCOL)
pkl.dump(clf_category_theme, open(os.path.join('models', 'classifiers', 'lvl2', 'clf_category_theme.pkl'), 'wb'),
         pkl.HIGHEST_PROTOCOL)
pkl.dump(clf_executor_category, open(os.path.join('models', 'classifiers', 'lvl2', 'clf_executor_category.pkl'), 'wb'),
         pkl.HIGHEST_PROTOCOL)
pkl.dump(clf_executor_theme, open(os.path.join('models', 'classifiers', 'lvl2', 'clf_executor_theme.pkl'), 'wb'),
         pkl.HIGHEST_PROTOCOL)
pkl.dump(clf_theme_category, open(os.path.join('models', 'classifiers', 'lvl2', 'clf_theme_category.pkl'), 'wb'),
         pkl.HIGHEST_PROTOCOL)
pkl.dump(clf_theme_executor, open(os.path.join('models', 'classifiers', 'lvl2', 'clf_theme_executor.pkl'), 'wb'),
         pkl.HIGHEST_PROTOCOL)

# Инициализируем модели третьего уровня
clf_category_executor_theme = LogisticRegression()
clf_executor_theme_category = LogisticRegression()
clf_category_theme_executor = LogisticRegression()

# Данные для моделей третьего уровня
X_train_category_executor, X_test_category_executor = pd.concat(
    [X_train, category_dummies_train, executor_dummies_train], axis=1), pd.concat(
    [X_test, category_dummies_test, executor_dummies_test], axis=1)

X_train_executor_theme, X_test_executor_theme = pd.concat(
    [X_train, executor_dummies_train, theme_dummies_train], axis=1), pd.concat(
    [X_test, executor_dummies_test, theme_dummies_test], axis=1)

X_train_category_theme, X_test_category_theme = pd.concat(
    [X_train, category_dummies_train, theme_dummies_train], axis=1), pd.concat(
    [X_test, category_dummies_test, theme_dummies_test], axis=1)

# Обучаем модели третьего уровня
clf_category_executor_theme.fit(X_train_category_executor, Y_theme_train)
clf_executor_theme_category.fit(X_train_executor_theme, Y_category_train)
clf_category_theme_executor.fit(X_train_category_theme, Y_executor_train)

# Определяем точность моделей третьего уровня
clf_category_executor_theme_accuracy = clf_category_executor_theme.score(X_test_category_executor, Y_theme_test)
clf_executor_theme_category_accuracy = clf_executor_theme_category.score(X_test_executor_theme, Y_category_test)
clf_category_theme_executor_accuracy = clf_category_theme_executor.score(X_test_category_theme, Y_executor_test)

model_accuracy.update({'clf_category_executor_theme': clf_category_executor_theme_accuracy})
model_accuracy.update({'clf_executor_theme_category': clf_executor_theme_category_accuracy})
model_accuracy.update({'clf_category_theme_executor': clf_category_theme_executor_accuracy})

print('Prediction accuracy of lvl3 models: \n'
      'clf_category_executor_theme = {0} \n'
      'clf_executor_theme_category = {1} \n'
      'clf_category_theme_executor = {2} \n'.format(clf_category_executor_theme_accuracy,
                                                    clf_executor_theme_category_accuracy,
                                                    clf_category_theme_executor_accuracy))

# Сохраняем модели третьего уровня
pkl.dump(clf_category_executor_theme,
         open(os.path.join('models', 'classifiers', 'lvl3', 'clf_category_executor_theme.pkl'), 'wb'),
         pkl.HIGHEST_PROTOCOL)
pkl.dump(clf_executor_theme_category,
         open(os.path.join('models', 'classifiers', 'lvl3', 'clf_executor_theme_category.pkl'), 'wb'),
         pkl.HIGHEST_PROTOCOL)
pkl.dump(clf_category_theme_executor,
         open(os.path.join('models', 'classifiers', 'lvl3', 'clf_category_theme_executor.pkl'), 'wb'),
         pkl.HIGHEST_PROTOCOL)

# Сохраняем коллекцию со значениями точности моделей
pkl.dump(model_accuracy, open(os.path.join('models', 'model_accuracy.pkl'), 'wb'),
         pkl.HIGHEST_PROTOCOL)
