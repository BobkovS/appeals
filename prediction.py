import os
import pickle as pkl

import pandas as pd

from utils import process_list

# Загружаем векторайзер
vectorizer = pkl.load(open(os.path.join('final_models', 'vectorizer.pkl'), 'rb'))

# Загружаем объекты LabelEncoder
category_label_encoder = pkl.load(
    open(os.path.join('final_models', 'label_encoders', 'category_label_encoder.pkl'), 'rb'))
executor_label_encoder = pkl.load(
    open(os.path.join('final_models', 'label_encoders', 'executor_label_encoder.pkl'), 'rb'))
theme_label_encoder = pkl.load(open(os.path.join('final_models', 'label_encoders', 'theme_label_encoder.pkl'), 'rb'))

# Загружаем модели первого уровня
clf_category = pkl.load(open(os.path.join('final_models', 'classifiers', 'lvl1', 'clf_category.pkl'), 'rb'))
clf_executor = pkl.load(open(os.path.join('final_models', 'classifiers', 'lvl1', 'clf_executor.pkl'), 'rb'))
clf_theme = pkl.load(open(os.path.join('final_models', 'classifiers', 'lvl1', 'clf_theme.pkl'), 'rb'))


def predict(input_phrase):
    prepared_phrase = ' '.join(process_list(input_phrase.split())).strip()
    prepared_phrase = pd.DataFrame(vectorizer.transform([prepared_phrase]).toarray())

    # Предсказываем с помощью моделей первого уровня
    clf_category_prediction = clf_category.predict(prepared_phrase)[0]
    clf_executor_prediction = clf_executor.predict(prepared_phrase)[0]
    clf_theme_prediction = clf_theme.predict(prepared_phrase)[0]

    theme_prediction = theme_label_encoder.inverse_transform([clf_theme_prediction])[0]
    executor_prediction = executor_label_encoder.inverse_transform([clf_executor_prediction])[0]
    category_prediction = category_label_encoder.inverse_transform([clf_category_prediction])[0]

    return category_prediction, executor_prediction, theme_prediction
