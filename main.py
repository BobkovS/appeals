import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

import nltk
nltk.download('wordnet')
from nltk.corpus import stopwords as sw
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import RussianStemmer
from nltk.stem import WordNetLemmatizer





stopwords = sw.words('russian')
en_stopwords = sw.words('english')
tokenizer = RegexpTokenizer(r'\w+')
morph = pymorphy2.MorphAnalyzer()
stemmer = RussianStemmer()
lemmatizer = WordNetLemmatizer()


data = pd.read_csv('primary_data.csv')