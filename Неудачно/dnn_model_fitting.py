import os
import pickle as pkl
import warnings

import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential, load_model
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import _tokenizer, process_list

np.random.seed = 42

warnings.filterwarnings("ignore")

# Загружаем данные
data = pd.read_csv(os.path.join('data', 'fully_prepared_data.csv'))

# Инициализируем и обучаем векторайзер
vectorizer = TfidfVectorizer(tokenizer=_tokenizer)
vectorizer.fit(data.text)
#pkl.dump(vectorizer, open(os.path.join('models', 'vectorizer.pkl'), 'wb'), pkl.HIGHEST_PROTOCOL)

# Инициализируем X и Y
X_train = vectorizer.transform(data.text)
X_train = pd.DataFrame(X_train.toarray())
Y_category_train = data.category.values
Y_executor_train = data.executor.values
Y_theme_train = data.theme.values

# Модель для категории
model = Sequential()
model.add(Dense(512, input_shape=(7124,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(len(np.unique(Y_category_train))))
model.add(Activation('softmax'))

checkpoint = ModelCheckpoint('dnn/model_weights/category/model-{epoch:03d}-{acc:03f}-{val_acc:03f}.h5', verbose=1,
                             monitor='val_loss',
                             save_best_only=True, mode='auto')

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_category_train, epochs=30, validation_split=0.1, callbacks=[checkpoint], verbose=True)

model.save(os.path.join('dnn', 'models', 'category.h5'))

# Модель для исполнителя
model = Sequential()
model.add(Dense(512, input_shape=(7124,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(len(np.unique(Y_executor_train))))
model.add(Activation('softmax'))

checkpoint = ModelCheckpoint('dnn/model_weights/executor/model-{epoch:03d}-{acc:03f}-{val_acc:03f}.h5', verbose=1,
                             monitor='val_loss',
                             save_best_only=True, mode='auto')

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_category_train, epochs=30, validation_split=0.1, callbacks=[checkpoint], verbose=True)

model.save(os.path.join('dnn', 'models', 'executor.h5'))

# Модель для темы
model = Sequential()
model.add(Dense(512, input_shape=(7124,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(len(np.unique(Y_theme_train))))
model.add(Activation('softmax'))

checkpoint = ModelCheckpoint('dnn/model_weights/theme/model-{epoch:03d}-{acc:03f}-{val_acc:03f}.h5', verbose=1,
                             monitor='val_loss',
                             save_best_only=True, mode='auto')

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_category_train, epochs=30, validation_split=0.1, callbacks=[checkpoint], verbose=True)

model.save(os.path.join('dnn', 'models', 'theme.h5'))

# Загружаем объекты LabelEncoder
category_label_encoder = pkl.load(
    open(os.path.join('final_models', 'label_encoders', 'category_label_encoder.pkl'), 'rb'))
executor_label_encoder = pkl.load(
    open(os.path.join('final_models', 'label_encoders', 'executor_label_encoder.pkl'), 'rb'))
theme_label_encoder = pkl.load(open(os.path.join('final_models', 'label_encoders', 'theme_label_encoder.pkl'), 'rb'))

dnn_category = load_model(os.path.join('dnn', 'models', 'category.h5'))
dnn_executor = load_model(os.path.join('dnn', 'models', 'executor.h5'))
dnn_theme = load_model(os.path.join('dnn', 'models', 'theme.h5'))


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