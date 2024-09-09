#Self Organized Map

#Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
dataset = pd.read_csv("data/Credit_Card_Applications.csv" )
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#Feature scalling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
x = sc.fit_transform(x)

#Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma=1.0, learning_rate=0.5)
som.random_weights_init(x)
som.train_random(data = x, num_iteration = 100)

#Visualing 
from pylab import bone,pcolor,colorbar,plot,show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ["o","s"]
colors = ["r","g"]
for i,X in enumerate(x):
    w = som.winner(X)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = "None",
         markersize = 10,
         markeredgewidth = 2)
show()

mappings = som.win_map(x)
frauds = np.concatenate((mappings[(8,1)],mappings[(6,8)]),axis=0)
frauds = sc.inverse_transform(frauds)