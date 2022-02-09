import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("./veriseti.csv")
x = dataset.iloc[:, :1].values
y = dataset.iloc[:, 1].values

lr_model = LinearRegression()
lr_model.fit(x,y)
te = [[18],[19],[20],[21]]
tahminSonuclari = lr_model.predict(te)
print(str(te) + " için sırasıyla tahmin sonuçları :" + str(tahminSonuclari))