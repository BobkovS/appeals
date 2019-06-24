import pandas as pd
data = pd.read_csv('data.csv')

data.drop(data[data.category=='Тестовая категория'].index, inplace=True)

print(data[data.category=='Туризм'].text)