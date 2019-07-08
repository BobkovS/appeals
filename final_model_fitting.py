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

# Инициализируем и обучаем векторайзер
vectorizer = TfidfVectorizer(tokenizer=_tokenizer)
vectorizer.fit(data.text)
pkl.dump(vectorizer, open(os.path.join('final_models', 'vectorizer.pkl'), 'wb'), pkl.HIGHEST_PROTOCOL)

# Инициализируем X и Y
X_train = vectorizer.transform(data.text)
X_train = pd.DataFrame(X_train.toarray())
Y_category_train = data.category
Y_executor_train = data.executor
Y_theme_train = data.theme

# Инициализируем модели
clf_category, clf_executor, clf_theme = LinearSVC(), LinearSVC(), LinearSVC()

# Учим модели первого уровня
clf_category.fit(X_train, Y_category_train)
clf_executor.fit(X_train, Y_executor_train)
clf_theme.fit(X_train, Y_theme_train)

# Сохраняем модели первого уровня
pkl.dump(clf_category, open(os.path.join('models', 'classifiers', 'lvl1', 'clf_category.pkl'), 'wb'),
         pkl.HIGHEST_PROTOCOL)
pkl.dump(clf_executor, open(os.path.join('models', 'classifiers', 'lvl1', 'clf_executor.pkl'), 'wb'),
         pkl.HIGHEST_PROTOCOL)
pkl.dump(clf_theme, open(os.path.join('models', 'classifiers', 'lvl1', 'clf_theme.pkl'), 'wb'), pkl.HIGHEST_PROTOCOL)
