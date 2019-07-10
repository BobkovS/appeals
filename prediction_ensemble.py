import os
import pickle as pkl
from collections import Counter

import pandas as pd

from utils import process_list

vectorizer = pkl.load(open(os.path.join('models', 'vectorizer.pkl'), 'rb'))

# Загружаем объекты LabelEncoder
category_label_encoder = pkl.load(
    open(os.path.join('models', 'label_encoders', 'category_label_encoder.pkl'), 'rb'))
executor_label_encoder = pkl.load(
    open(os.path.join('models', 'label_encoders', 'executor_label_encoder.pkl'), 'rb'))
theme_label_encoder = pkl.load(open(os.path.join('models', 'label_encoders', 'theme_label_encoder.pkl'), 'rb'))

# Загружаем модели
clf_svc_category = pkl.load(open(os.path.join('models', 'classifiers', 'lvl1', 'clf_svc_category.pkl'), 'rb'))
clf_pac_category = pkl.load(open(os.path.join('models', 'classifiers', 'lvl1', 'clf_pac_category.pkl'), 'rb'))
clf_ridge_category = pkl.load(open(os.path.join('models', 'classifiers', 'lvl1', 'clf_ridge_category.pkl'), 'rb'))
clf_sgdc_category = pkl.load(open(os.path.join('models', 'classifiers', 'lvl1', 'clf_sgdc_category.pkl'), 'rb'))
clf_etc_category = pkl.load(open(os.path.join('models', 'classifiers', 'lvl1', 'clf_etc_category.pkl'), 'rb'))

clf_svc_executor = pkl.load(open(os.path.join('models', 'classifiers', 'lvl1', 'clf_svc_executor.pkl'), 'rb'))
clf_pac_executor = pkl.load(open(os.path.join('models', 'classifiers', 'lvl1', 'clf_pac_executor.pkl'), 'rb'))
clf_ridge_executor = pkl.load(open(os.path.join('models', 'classifiers', 'lvl1', 'clf_ridge_executor.pkl'), 'rb'))
clf_sgdc_executor = pkl.load(open(os.path.join('models', 'classifiers', 'lvl1', 'clf_sgdc_executor.pkl'), 'rb'))
clf_etc_executor = pkl.load(open(os.path.join('models', 'classifiers', 'lvl1', 'clf_etc_executor.pkl'), 'rb'))

clf_svc_theme = pkl.load(open(os.path.join('models', 'classifiers', 'lvl1', 'clf_svc_theme.pkl'), 'rb'))
clf_pac_theme = pkl.load(open(os.path.join('models', 'classifiers', 'lvl1', 'clf_pac_theme.pkl'), 'rb'))
clf_ridge_theme = pkl.load(open(os.path.join('models', 'classifiers', 'lvl1', 'clf_ridge_theme.pkl'), 'rb'))
clf_sgdc_theme = pkl.load(open(os.path.join('models', 'classifiers', 'lvl1', 'clf_sgdc_theme.pkl'), 'rb'))
clf_etc_theme = pkl.load(open(os.path.join('models', 'classifiers', 'lvl1', 'clf_etc_theme.pkl'), 'rb'))


def predict(input_phrase):
    prepared_phrase = ' '.join(process_list(input_phrase.split())).strip()
    prepared_phrase = pd.DataFrame(vectorizer.transform([prepared_phrase]).toarray())

    # Получаем предсказания моделей
    clf_svc_category_prediction = clf_svc_category.predict(prepared_phrase)[0]
    clf_pac_category_prediction = clf_pac_category.predict(prepared_phrase)[0]
    clf_ridge_category_prediction = clf_ridge_category.predict(prepared_phrase)[0]
    clf_sgdc_category_prediction = clf_sgdc_category.predict(prepared_phrase)[0]
    clf_etc_category_prediction = clf_etc_category.predict(prepared_phrase)[0]

    clf_svc_executor_prediction = clf_svc_executor.predict(prepared_phrase)[0]
    clf_pac_executor_prediction = clf_pac_executor.predict(prepared_phrase)[0]
    clf_ridge_executor_prediction = clf_ridge_executor.predict(prepared_phrase)[0]
    clf_sgdc_executor_prediction = clf_sgdc_executor.predict(prepared_phrase)[0]
    clf_etc_executor_prediction = clf_etc_executor.predict(prepared_phrase)[0]

    clf_svc_theme_prediction = clf_svc_theme.predict(prepared_phrase)[0]
    clf_pac_theme_prediction = clf_pac_theme.predict(prepared_phrase)[0]
    clf_ridge_theme_prediction = clf_ridge_theme.predict(prepared_phrase)[0]
    clf_sgdc_theme_prediction = clf_sgdc_theme.predict(prepared_phrase)[0]
    clf_etc_theme_prediction = clf_etc_theme.predict(prepared_phrase)[0]

    # Методом голосования получаем общий предикт

    predictions = [clf_svc_category_prediction, clf_pac_category_prediction, clf_ridge_category_prediction,
                   clf_sgdc_category_prediction, clf_etc_category_prediction]
    final_predictions_category = (Counter(predictions).most_common(1)[0][0])

    predictions = [clf_svc_executor_prediction, clf_pac_executor_prediction, clf_ridge_executor_prediction,
                   clf_sgdc_executor_prediction, clf_etc_executor_prediction]
    final_predictions_executor = (Counter(predictions).most_common(1)[0][0])

    predictions = [clf_svc_theme_prediction, clf_pac_theme_prediction, clf_ridge_theme_prediction,
                   clf_sgdc_theme_prediction, clf_etc_theme_prediction]
    final_predictions_theme = (Counter(predictions).most_common(1)[0][0])

    theme_prediction = theme_label_encoder.inverse_transform([final_predictions_theme])[0]
    executor_prediction = executor_label_encoder.inverse_transform([final_predictions_executor])[0]
    category_prediction = category_label_encoder.inverse_transform([final_predictions_category])[0]

    return category_prediction, executor_prediction, theme_prediction
