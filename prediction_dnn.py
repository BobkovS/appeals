import os
import pickle as pkl

import pandas as pd
from keras.models import load_model

from utils import process_list

# Загружаем объекты LabelEncoder
category_label_encoder = pkl.load(
    open(os.path.join('final_models', 'label_encoders', 'category_label_encoder.pkl'), 'rb'))
executor_label_encoder = pkl.load(
    open(os.path.join('final_models', 'label_encoders', 'executor_label_encoder.pkl'), 'rb'))
theme_label_encoder = pkl.load(open(os.path.join('final_models', 'label_encoders', 'theme_label_encoder.pkl'), 'rb'))

dnn_category = load_model(os.path.join('final_models_dnn', 'models', 'category.h5'))
dnn_executor = load_model(os.path.join('final_models_dnn', 'models', 'executor.h5'))
dnn_theme = load_model(os.path.join('final_models_dnn', 'models', 'theme.h5'))

vectorizer = pkl.load(open(os.path.join('final_models_dnn', 'vectorizer.pkl'), 'rb'))


def predict_with_dnn(input_phrase):
    prepared_phrase = ' '.join(process_list(input_phrase.split())).strip()
    prepared_phrase = pd.DataFrame(vectorizer.transform([prepared_phrase]).toarray())

    # Предсказываем с помощью моделей первого уровня
    clf_category_prediction = dnn_category.predict_classes(prepared_phrase)[0]
    clf_executor_prediction = dnn_executor.predict_classes(prepared_phrase)[0]
    clf_theme_prediction = dnn_theme.predict_classes(prepared_phrase)[0]

    theme_prediction = theme_label_encoder.inverse_transform([clf_theme_prediction])[0]
    executor_prediction = executor_label_encoder.inverse_transform([clf_executor_prediction])[0]
    category_prediction = category_label_encoder.inverse_transform([clf_category_prediction])[0]

    return category_prediction, executor_prediction, theme_prediction


print(predict_with_dnn('У меня не работает сфетофор'))
