# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import dataset
dataset = pd.read_csv('https://raw.githubusercontent.com/ankishore/som_fraud_detection/main/Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Train SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, # 10x10 SOM grid
              input_len = 15, # 15 columns/features in X
              sigma = 1.0, # Neighborhood radius. Scaled between 0 - 1
              learning_rate = 0.5) # Rate at which weights are updated
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualize results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T) # Mean distances of all winning nodes
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x) # Get winning node of customer x
    plot(w[0] + 0.5, # X co-ordinate of winning node
         w[1] + 0.5, # Y co-ordinate. 0.5 to put marker in middle
         markers[Y[i]],
         markeredgecolor = colors[Y[i]],
         markerfacecolor = 'None', # Don't color inside the marker
         markersize = 10,
         markeredgewidth = 2)
show()

# Finding the fraud
mappings = som.win_map(X) # Get all mappings from winning node to customer key (X, Y) co-ordinates
frauds = np.concatenate((mappings[(8, 1)], mappings[(4, 8)], mappings[2, 5]), axis = 0)
frauds = sc.inverse_transform(frauds)
for i in frauds[:, 0]:
    print(int(i))