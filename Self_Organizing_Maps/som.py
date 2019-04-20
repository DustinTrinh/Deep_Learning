# Self Organizing Map

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
#first : is select all rows, :-1 means take all column but not last one
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Feature scaling using normalization
sc = MinMaxScaler(feature_range = (0,1))
X = sc.fit_transform(X)

som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 150 )

#Visualizing the results
from pylab import bone,pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
#Red -> No Approval / Green -> Approval
markers = ['o', 's']
colors = ['r', 'g']

for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5, 
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors [y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

#Finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(8,2)], mappings[(5,8)]), axis = 0)
frauds = sc.inverse_transform(frauds)