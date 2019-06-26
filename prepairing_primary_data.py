import os
import pickle as pkl

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from utils import process_list

# Загрузка данных
data = pd.read_csv(os.path.join('data', 'primary_data.csv'))

# Удаление строк с пустыми параметрами
data = data.dropna()

# Удаление строчки с значениями атрибутов ["Тестовая категория", "Тестовый департамент"...]
data.drop(data[data.category == 'Тестовая категория'].index, inplace=True)
data = data.reset_index(drop=True)

# Преобразование записей в колонках ['Category', 'Executor', 'Theme'] к виду
# (проблемы_дорожного_покрытия) вместо (проблемы дорожного покрытия)
for index, row in data.iterrows():
    data.category[index] = data.category[index].strip().replace(' ', '_')
    data.executor[index] = data.executor[index].strip().replace(' ', '_')
    data.theme[index] = data.theme[index].strip().replace(' ', '_')

# Преобразование значений в колонке ['Text'] к приемлемой форме (Удаление стоп слов, приведение к нормальной форме,
# удаление цифр, удаление иностранных символов)
for index, row in data.iterrows():
    new_text = process_list(data.text[index].split())
    data.text[index] = ' '.join(new_text)

# Сохранение данных
data.to_csv(os.path.join('data', 'prepared_data.csv'), index=False)

# Инициализация объектов LabelEncoder
category_label_encoder, executor_label_encoder, theme_label_encoder = LabelEncoder(), LabelEncoder(), LabelEncoder()

# Применение LabelEncoder к колонкам ['Category', 'Executor', 'Theme'] и сохранение энкодеров в системе
data.category = category_label_encoder.fit_transform(data.category)
pkl.dump(category_label_encoder, open(os.path.join('models', 'label_encoders', 'category_label_encoder.pkl'), 'wb'),
         pkl.HIGHEST_PROTOCOL)

data.executor = executor_label_encoder.fit_transform(data.executor)
pkl.dump(executor_label_encoder, open(os.path.join('models', 'label_encoders', 'executor_label_encoder.pkl'), 'wb'),
         pkl.HIGHEST_PROTOCOL)

data.theme = theme_label_encoder.fit_transform(data.theme)
pkl.dump(theme_label_encoder, open(os.path.join('models', 'label_encoders', 'theme_label_encoder.pkl'), 'wb'),
         pkl.HIGHEST_PROTOCOL)

# Сохранение данных
data.to_csv(os.path.join('data', 'fully_prepared_data.csv'), index=False)
