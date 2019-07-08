import os
import pickle as pkl

import pandas as pd

from utils import process_list

vectorizer = pkl.load(open(os.path.join('models', 'vectorizer.pkl'), 'rb'))

test_data = pd.read_csv(os.path.join('data', 'test_data.csv'))
length_of_test_data = len(test_data)

count_of_correct_categories = 0
count_of_correct_executors = 0
count_of_correct_themes = 0

for index, row in test_data.iterrows():
    category_predictions, executor_predictions, theme_predictions = [], [], []

    input_phrase = row.text

    prepared_phrase = ' '.join(process_list(input_phrase.split())).strip()
    prepared_phrase = pd.DataFrame(vectorizer.transform([prepared_phrase]).toarray())

    # Загружаем объекты LabelEncoder
    category_label_encoder = pkl.load(
        open(os.path.join('models', 'label_encoders', 'category_label_encoder.pkl'), 'rb'))
    executor_label_encoder = pkl.load(
        open(os.path.join('models', 'label_encoders', 'executor_label_encoder.pkl'), 'rb'))
    theme_label_encoder = pkl.load(open(os.path.join('models', 'label_encoders', 'theme_label_encoder.pkl'), 'rb'))

    # Предсказываем с помощью моделей первого уровня
    clf_category = pkl.load(open(os.path.join('models', 'classifiers', 'lvl1', 'clf_category.pkl'), 'rb'))
    clf_executor = pkl.load(open(os.path.join('models', 'classifiers', 'lvl1', 'clf_executor.pkl'), 'rb'))
    clf_theme = pkl.load(open(os.path.join('models', 'classifiers', 'lvl1', 'clf_theme.pkl'), 'rb'))

    clf_category_prediction = clf_category.predict(prepared_phrase)[0]
    clf_executor_prediction = clf_executor.predict(prepared_phrase)[0]
    clf_theme_prediction = clf_theme.predict(prepared_phrase)[0]

    if clf_theme_prediction == row.theme: count_of_correct_themes += 1
    if clf_executor_prediction == row.executor: count_of_correct_executors += 1
    if clf_category_prediction == row.category: count_of_correct_categories += 1

theme_prediction_accuracy = count_of_correct_themes / length_of_test_data
executor_prediction_accuracy = count_of_correct_executors / length_of_test_data
category_prediction_accuracy = count_of_correct_categories / length_of_test_data

print('Prediction accuracy for Category = {0}, Executor = {1}, Theme = {2}'.format(category_prediction_accuracy,
                                                                                   executor_prediction_accuracy,
                                                                                   theme_prediction_accuracy))
