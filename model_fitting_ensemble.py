import os
import pickle as pkl
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier as Etc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier as Pac, RidgeClassifier, SGDClassifier
from sklearn.svm import LinearSVC

from utils import _tokenizer

np.random.seed = 42

warnings.filterwarnings("ignore")

# Загружаем данные
data = pd.read_csv(os.path.join('data', 'fully_prepared_data.csv'))

# Инициализируем и обучаем векторайзер
vectorizer = TfidfVectorizer(tokenizer=_tokenizer)
vectorizer.fit(data.text)
pkl.dump(vectorizer, open(os.path.join('final_models_ensemble', 'vectorizer.pkl'), 'wb'), pkl.HIGHEST_PROTOCOL)

# Инициализируем X и Y
X_train = vectorizer.transform(data.text)
X_train = pd.DataFrame(X_train.toarray())
Y_category_train = data.category
Y_executor_train = data.executor
Y_theme_train = data.theme

# Инициализируем модели
clf_svc_category, clf_svc_executor, clf_svc_theme = LinearSVC(), LinearSVC(), LinearSVC()
clf_pac_category, clf_pac_executor, clf_pac_theme = Pac(), Pac(), Pac()
clf_ridge_category, clf_ridge_executor, clf_ridge_theme = RidgeClassifier(), RidgeClassifier(), RidgeClassifier()
clf_sgdc_category, clf_sgdc_executor, clf_sgdc_theme = SGDClassifier(), SGDClassifier(), SGDClassifier()
clf_etc_category, clf_etc_executor, clf_etc_theme = Etc(), Etc(), Etc()

# Учим модели
clf_svc_category.fit(X_train, Y_category_train)
clf_pac_category.fit(X_train, Y_category_train)
clf_ridge_category.fit(X_train, Y_category_train)
clf_sgdc_category.fit(X_train, Y_category_train)
clf_etc_category.fit(X_train, Y_category_train)

clf_svc_executor.fit(X_train, Y_executor_train)
clf_pac_executor.fit(X_train, Y_executor_train)
clf_ridge_executor.fit(X_train, Y_executor_train)
clf_sgdc_executor.fit(X_train, Y_executor_train)
clf_etc_executor.fit(X_train, Y_executor_train)

clf_svc_theme.fit(X_train, Y_theme_train)
clf_pac_theme.fit(X_train, Y_theme_train)
clf_ridge_theme.fit(X_train, Y_theme_train)
clf_sgdc_theme.fit(X_train, Y_theme_train)
clf_etc_theme.fit(X_train, Y_theme_train)

# Сохраняем модели первого уровня
pkl.dump(clf_svc_category,
         open(os.path.join('final_models_ensemble', 'classifiers', 'lvl1', 'clf_svc_category.pkl'), 'wb'),
         pkl.HIGHEST_PROTOCOL)
pkl.dump(clf_pac_category,
         open(os.path.join('final_models_ensemble', 'classifiers', 'lvl1', 'clf_pac_category.pkl'), 'wb'),
         pkl.HIGHEST_PROTOCOL)
pkl.dump(clf_ridge_category,
         open(os.path.join('final_models_ensemble', 'classifiers', 'lvl1', 'clf_ridge_category.pkl'), 'wb'),
         pkl.HIGHEST_PROTOCOL)
pkl.dump(clf_sgdc_category,
         open(os.path.join('final_models_ensemble', 'classifiers', 'lvl1', 'clf_sgdc_category.pkl'), 'wb'),
         pkl.HIGHEST_PROTOCOL)
pkl.dump(clf_etc_category,
         open(os.path.join('final_models_ensemble', 'classifiers', 'lvl1', 'clf_etc_category.pkl'), 'wb'),
         pkl.HIGHEST_PROTOCOL)
pkl.dump(clf_svc_executor,
         open(os.path.join('final_models_ensemble', 'classifiers', 'lvl1', 'clf_svc_executor.pkl'), 'wb'),
         pkl.HIGHEST_PROTOCOL)
pkl.dump(clf_pac_executor,
         open(os.path.join('final_models_ensemble', 'classifiers', 'lvl1', 'clf_pac_executor.pkl'), 'wb'),
         pkl.HIGHEST_PROTOCOL)
pkl.dump(clf_ridge_executor,
         open(os.path.join('final_models_ensemble', 'classifiers', 'lvl1', 'clf_ridge_executor.pkl'), 'wb'),
         pkl.HIGHEST_PROTOCOL)
pkl.dump(clf_sgdc_executor,
         open(os.path.join('final_models_ensemble', 'classifiers', 'lvl1', 'clf_sgdc_executor.pkl'), 'wb'),
         pkl.HIGHEST_PROTOCOL)
pkl.dump(clf_etc_executor,
         open(os.path.join('final_models_ensemble', 'classifiers', 'lvl1', 'clf_etc_executor.pkl'), 'wb'),
         pkl.HIGHEST_PROTOCOL)
pkl.dump(clf_svc_theme, open(os.path.join('final_models_ensemble', 'classifiers', 'lvl1', 'clf_svc_theme.pkl'), 'wb'),
         pkl.HIGHEST_PROTOCOL)
pkl.dump(clf_pac_theme, open(os.path.join('final_models_ensemble', 'classifiers', 'lvl1', 'clf_pac_theme.pkl'), 'wb'),
         pkl.HIGHEST_PROTOCOL)
pkl.dump(clf_ridge_theme,
         open(os.path.join('final_models_ensemble', 'classifiers', 'lvl1', 'clf_ridge_theme.pkl'), 'wb'),
         pkl.HIGHEST_PROTOCOL)
pkl.dump(clf_sgdc_theme, open(os.path.join('final_models_ensemble', 'classifiers', 'lvl1', 'clf_sgdc_theme.pkl'), 'wb'),
         pkl.HIGHEST_PROTOCOL)
pkl.dump(clf_etc_theme, open(os.path.join('final_models_ensemble', 'classifiers', 'lvl1', 'clf_etc_theme.pkl'), 'wb'),
         pkl.HIGHEST_PROTOCOL)
