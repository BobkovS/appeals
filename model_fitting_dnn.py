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
pkl.dump(vectorizer, open(os.path.join('final_models_dnn', 'models', 'vectorizer.pkl'), 'wb'), pkl.HIGHEST_PROTOCOL)

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

checkpoint = ModelCheckpoint('final_models_dnn/model_weights/category/model-{epoch:03d}-{acc:03f}-{val_acc:03f}.h5', verbose=1,
                             monitor='val_loss',
                             save_best_only=True, mode='auto')

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_category_train, epochs=30, validation_split=0.1, callbacks=[checkpoint], verbose=True)

model.save(os.path.join('final_models_dnn', 'models', 'category.h5'))

# Модель для исполнителя
model = Sequential()
model.add(Dense(512, input_shape=(7124,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(len(np.unique(Y_executor_train))))
model.add(Activation('softmax'))

checkpoint = ModelCheckpoint('final_models_dnn/model_weights/executor/model-{epoch:03d}-{acc:03f}-{val_acc:03f}.h5', verbose=1,
                             monitor='val_loss',
                             save_best_only=True, mode='auto')

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_executor_train, epochs=30, validation_split=0.1, callbacks=[checkpoint], verbose=True)

model.save(os.path.join('final_models_dnn', 'models', 'executor.h5'))

# Модель для темы
model = Sequential()
model.add(Dense(512, input_shape=(7124,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(len(np.unique(Y_theme_train))))
model.add(Activation('softmax'))

checkpoint = ModelCheckpoint('final_models_dnn/model_weights/theme/model-{epoch:03d}-{acc:03f}-{val_acc:03f}.h5', verbose=1,
                             monitor='val_loss',
                             save_best_only=True, mode='auto')

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_theme_train, epochs=30, validation_split=0.1, callbacks=[checkpoint], verbose=True)

model.save(os.path.join('final_models_dnn', 'models', 'theme.h5'))