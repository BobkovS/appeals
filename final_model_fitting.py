import os
import pickle as pkl
import warnings

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from utils import _tokenizer

warnings.filterwarnings("ignore")

# Загружаем данные
data = pd.read_csv(os.path.join('data', 'fully_prepared_data.csv'))

# Отдельно запоминаем мешки слов для столбцов [Category, Executor, Theme]
category_dummies = pd.get_dummies(data.category, prefix='category')
pkl.dump(category_dummies.columns, open(os.path.join('final_models', 'dummies_columns', 'category_columns.pkl'), 'wb'),
         pkl.HIGHEST_PROTOCOL)

executor_dummies = pd.get_dummies(data.executor, prefix='executor')
pkl.dump(executor_dummies.columns, open(os.path.join('final_models', 'dummies_columns', 'executor_columns.pkl'), 'wb'),
         pkl.HIGHEST_PROTOCOL)

theme_dummies = pd.get_dummies(data.theme, prefix='theme')
pkl.dump(theme_dummies.columns, open(os.path.join('final_models', 'dummies_columns', 'theme_columns.pkl'), 'wb'),
         pkl.HIGHEST_PROTOCOL)

# Инициализируем и обучаем векторайзер
vectorizer = CountVectorizer(tokenizer=_tokenizer)
vectorizer.fit(data.text)
pkl.dump(vectorizer, open(os.path.join('final_models', 'vectorizer.pkl'), 'wb'), pkl.HIGHEST_PROTOCOL)

# Инициализируем X и Y
X = vectorizer.transform(data.text)
X = pd.DataFrame(X.toarray())
Y_category = data.category
Y_executor = data.executor
Y_theme = data.theme

# Инициализируем объекты XGBClassifier
clf_category, clf_executor, clf_theme = LogisticRegression(), LogisticRegression(), LogisticRegression()

# Учим модели первого уровня
clf_category.fit(X, Y_category)
clf_executor.fit(X, Y_executor)
clf_theme.fit(X, Y_theme)

# Сохраняем модели первого уровня
pkl.dump(clf_category, open(os.path.join('final_models', 'classifiers', 'lvl1', 'clf_category.pkl'), 'wb'),
         pkl.HIGHEST_PROTOCOL)
pkl.dump(clf_executor, open(os.path.join('final_models', 'classifiers', 'lvl1', 'clf_executor.pkl'), 'wb'),
         pkl.HIGHEST_PROTOCOL)
pkl.dump(clf_theme, open(os.path.join('final_models', 'classifiers', 'lvl1', 'clf_theme.pkl'), 'wb'),
         pkl.HIGHEST_PROTOCOL)

# Данные для моделей второго уровня
X_category = pd.concat([X, category_dummies], axis=1)
X_executor = pd.concat([X, executor_dummies], axis=1)
X_theme = pd.concat([X, theme_dummies], axis=1)

# Инициализируем модели второго уровня
clf_category_executor, clf_category_theme = LogisticRegression(), LogisticRegression()

clf_executor_category, clf_executor_theme = LogisticRegression(), LogisticRegression()

clf_theme_category, clf_theme_executor = LogisticRegression(), LogisticRegression()

# Обучаем модели второго уровня
clf_category_executor.fit(X_category, Y_executor)
clf_category_theme.fit(X_category, Y_theme)

clf_executor_category.fit(X_executor, Y_category)
clf_executor_theme.fit(X_executor, Y_theme)

clf_theme_category.fit(X_theme, Y_category)
clf_theme_executor.fit(X_theme, Y_executor)

# Сохраняем модели второго уровня
pkl.dump(clf_category_executor,
         open(os.path.join('final_models', 'classifiers', 'lvl2', 'clf_category_executor.pkl'), 'wb'),
         pkl.HIGHEST_PROTOCOL)
pkl.dump(clf_category_theme, open(os.path.join('final_models', 'classifiers', 'lvl2', 'clf_category_theme.pkl'), 'wb'),
         pkl.HIGHEST_PROTOCOL)
pkl.dump(clf_executor_category,
         open(os.path.join('final_models', 'classifiers', 'lvl2', 'clf_executor_category.pkl'), 'wb'),
         pkl.HIGHEST_PROTOCOL)
pkl.dump(clf_executor_theme, open(os.path.join('final_models', 'classifiers', 'lvl2', 'clf_executor_theme.pkl'), 'wb'),
         pkl.HIGHEST_PROTOCOL)
pkl.dump(clf_theme_category, open(os.path.join('final_models', 'classifiers', 'lvl2', 'clf_theme_category.pkl'), 'wb'),
         pkl.HIGHEST_PROTOCOL)
pkl.dump(clf_theme_executor, open(os.path.join('final_models', 'classifiers', 'lvl2', 'clf_theme_executor.pkl'), 'wb'),
         pkl.HIGHEST_PROTOCOL)

# Инициализируем модели третьего уровня
clf_category_executor_theme = LogisticRegression()
clf_executor_theme_category = LogisticRegression()
clf_category_theme_executor = LogisticRegression()

# Данные для моделей третьего уровня
X_category_executor = pd.concat(
    [X, category_dummies, executor_dummies], axis=1)

X_executor_theme = pd.concat(
    [X, executor_dummies, theme_dummies], axis=1)

X_category_theme = pd.concat(
    [X, category_dummies, theme_dummies], axis=1)

# Обучаем модели третьего уровня
clf_category_executor_theme.fit(X_category_executor, Y_theme)
clf_executor_theme_category.fit(X_executor_theme, Y_category)
clf_category_theme_executor.fit(X_category_theme, Y_executor)

# Сохраняем модели третьего уровня
pkl.dump(clf_category_executor_theme,
         open(os.path.join('final_models', 'classifiers', 'lvl3', 'clf_category_executor_theme.pkl'), 'wb'),
         pkl.HIGHEST_PROTOCOL)
pkl.dump(clf_executor_theme_category,
         open(os.path.join('final_models', 'classifiers', 'lvl3', 'clf_executor_theme_category.pkl'), 'wb'),
         pkl.HIGHEST_PROTOCOL)
pkl.dump(clf_category_theme_executor,
         open(os.path.join('final_models', 'classifiers', 'lvl3', 'clf_category_theme_executor.pkl'), 'wb'),
         pkl.HIGHEST_PROTOCOL)
