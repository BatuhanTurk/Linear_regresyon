import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import re
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pickle

def optimizasyon(dataset):
    dataset = dataset.dropna()

    stop_words = set(stopwords.words('turkish'))
    noktalamaIsaretleri = ['•', '!', '"', '#', '”'
    , '“', '$', '%', '&', "'", '–', '(', ')', '*'
    , '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[',
     '\\', ']', '^', '_', '`', '{', '|', '}', '~', '0', '1', '2', '3',
      '4', '5', '6', '7', '8', '9', '…']
    stop_words.update(noktalamaIsaretleri)

    for ind in dataset.index:
        body = dataset['Body'][ind]
        body = body.lower()
        body = re.sub(r'http\S+', '', body)
        body = re.sub("\[[^]]*\]", '', body)
        body = (" ").join([word for word in body.split() if not word in stop_words])
        body = "".join([char for char in body if not char in noktalamaIsaretleri])
        dataset['Body'][ind] = body
    return dataset

nltk.download('stopwords')
dataset = pd.read_csv('dataset.csv')
dataset.sort_values("Body", inplace = True)
dataset = dataset.drop(columns="B")
dataset.drop_duplicates(subset ="Body",keep = False, inplace = True)
dataset = optimizasyon(dataset)

yorumlar_makina = dataset[dataset['Label']==0]
yorumlar_insan = dataset[dataset['Label']==1]

tfIdf = TfidfVectorizer( binary=False, ngram_range=(1,3))

makina_vec = tfIdf.fit_transform(yorumlar_makina['Body'].tolist())
insan_vec = tfIdf.fit_transform(yorumlar_insan['Body'].tolist())

x = dataset['Body']
y = dataset['Label']

x_vec = tfIdf.fit_transform(X)

x_egitim_vec, x_test_vec, y_egitim, y_test = train_test_split(x_vec, y, test_size=0.2, random_state=0)

lojistikRegresyon = LogisticRegression() 
lojistikRegresyon.fit(x_egitim_vec,y_egitim)
y_tahmin = lojistikRegresyon.predict(x_test_vec)

pickle.dump(lojistikRegresyon, open("egitilmis_model", 'wb'))
print("Lojistik Regresyon modeli eğitildi ve kayıt edildi !")

pickle.dump(tfIdf, open("vektorlestirici", 'wb'))
print("Tf-Idf vektörleştirici modeli kayıt edildi !")