import os
import pickle as pkl

import numpy as np
import pandas as pd

from utils import process_list, match_value_with_dummies

from collections import Counter

category_predictions, executor_predictions, theme_predictions = [], [], []

vectorizer = pkl.load(open(os.path.join('models', 'vectorizer.pkl'), 'rb'))

input_phrase = 'Приезжал к знакомому на ул. Шевченко 36/50 (2-я Линия 50), он мне решил пожаловаться на отсутствие' \
               ' освещения во дворе и прилегающей территории. Я был удивлён когда наступил вечер и уходил от ' \
               'него с включенным фонариком на телефоне (ну вообще ничего не видно). Очень много мелких аварийных ' \
               'ситуаций с автомобилями, не помогают даже включенные фары. Особенно неудобно в период конца лета, ' \
               'осени и зимой, когда на улице темнеет рано и требуется освещение + густая растительность загораживает' \
               ' в вечерний период лучи солнца, когда ещё достаточно свело. Вырубать деревья конечно не нужно, ' \
               'требуется не это. Огромная просьба от жильцов и от меня сделать уличное освещение. Даже 3-4 фонаря' \
               ' улучшат обстановку. Рядом всё же и РязГМУ, там и студенты ходят, центр города. Я надеюсь' \
               ' на скорейший и положительный ответ.'

prepared_phrase = ' '.join(process_list(input_phrase.split())).strip()
prepared_phrase = pd.DataFrame(vectorizer.transform([prepared_phrase]).toarray())

# Загружаем объекты LabelEncoder
category_label_encoder = pkl.load(open(os.path.join('models', 'label_encoders', 'category_label_encoder.pkl'), 'rb'))
executor_label_encoder = pkl.load(open(os.path.join('models', 'label_encoders', 'executor_label_encoder.pkl'), 'rb'))
theme_label_encoder = pkl.load(open(os.path.join('models', 'label_encoders', 'theme_label_encoder.pkl'), 'rb'))

# Предсказываем с помощью моделей первого уровня
clf_category = pkl.load(open(os.path.join('models', 'classifiers', 'lvl1', 'clf_category.pkl'), 'rb'))
clf_executor = pkl.load(open(os.path.join('models', 'classifiers', 'lvl1', 'clf_executor.pkl'), 'rb'))
clf_theme = pkl.load(open(os.path.join('models', 'classifiers', 'lvl1', 'clf_theme.pkl'), 'rb'))

clf_category_prediction = clf_category.predict(prepared_phrase)[0]
clf_executor_prediction = clf_executor.predict(prepared_phrase)[0]
clf_theme_prediction = clf_theme.predict(prepared_phrase)[0]

category_predictions.append(clf_category_prediction)
executor_predictions.append(clf_executor_prediction)
theme_predictions.append(clf_theme_prediction)

# Предсказываем с помощью моделей второго уровня и предсказаний моделей первого уровня

category_dummies_columns = pkl.load(open(os.path.join('models', 'dummies_columns', 'category_columns.pkl'), 'rb'))
executor_dummies_columns = pkl.load(open(os.path.join('models', 'dummies_columns', 'executor_columns.pkl'), 'rb'))
theme_dummies_columns = pkl.load(open(os.path.join('models', 'dummies_columns', 'theme_columns.pkl'), 'rb'))

category_dummies = pd.DataFrame(0, index=np.arange(1), columns=category_dummies_columns)
executor_dummies = pd.DataFrame(0, index=np.arange(1), columns=executor_dummies_columns)
theme_dummies = pd.DataFrame(0, index=np.arange(1), columns=theme_dummies_columns)

prepared_phrase_with_category = pd.concat([prepared_phrase, category_dummies], axis=1)
prepared_phrase_with_executor = pd.concat([prepared_phrase, executor_dummies], axis=1)
prepared_phrase_with_theme = pd.concat([prepared_phrase, theme_dummies], axis=1)

prepared_phrase_with_category = match_value_with_dummies(clf_category_prediction, prepared_phrase_with_category,
                                                         'category')
prepared_phrase_with_executor = match_value_with_dummies(clf_executor_prediction, prepared_phrase_with_executor,
                                                         'executor')
prepared_phrase_with_theme = match_value_with_dummies(clf_theme_prediction, prepared_phrase_with_theme, 'theme')

clf_category_executor = pkl.load(open(os.path.join('models', 'classifiers', 'lvl2', 'clf_category_executor.pkl'), 'rb'))
clf_category_theme = pkl.load(open(os.path.join('models', 'classifiers', 'lvl2', 'clf_category_theme.pkl'), 'rb'))
clf_executor_category = pkl.load(open(os.path.join('models', 'classifiers', 'lvl2', 'clf_executor_category.pkl'), 'rb'))
clf_executor_theme = pkl.load(open(os.path.join('models', 'classifiers', 'lvl2', 'clf_executor_theme.pkl'), 'rb'))
clf_theme_category = pkl.load(open(os.path.join('models', 'classifiers', 'lvl2', 'clf_theme_category.pkl'), 'rb'))
clf_theme_executor = pkl.load(open(os.path.join('models', 'classifiers', 'lvl2', 'clf_theme_executor.pkl'), 'rb'))

clf_category_executor_prediction = clf_category_executor.predict(prepared_phrase_with_category)[0]
clf_category_theme_prediction = clf_category_theme.predict(prepared_phrase_with_category)[0]
clf_executor_category_prediction = clf_executor_category.predict(prepared_phrase_with_executor)[0]
clf_executor_theme_prediction = clf_executor_theme.predict(prepared_phrase_with_executor)[0]
clf_theme_category_prediction = clf_theme_category.predict(prepared_phrase_with_theme)[0]
clf_theme_executor_prediction = clf_theme_executor.predict(prepared_phrase_with_theme)[0]

category_predictions.append(clf_executor_category_prediction)
category_predictions.append(clf_theme_category_prediction)
executor_predictions.append(clf_category_executor_prediction)
executor_predictions.append(clf_theme_executor_prediction)
theme_predictions.append(clf_category_theme_prediction)
theme_predictions.append(clf_executor_theme_prediction)

print(Counter(category_predictions).most_common(1))
