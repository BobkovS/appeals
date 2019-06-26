import os
import pickle as pkl

import pandas as pd

from utils import process_list

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

clf_category_prediction = clf_category.predict(prepared_phrase)
clf_executor_prediction = clf_executor.predict(prepared_phrase)
clf_theme_prediction = clf_theme.predict(prepared_phrase)


