import numpy as np
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from pyearth import Earth
from demodata import DemoData
    
# Create some fake data
if __name__ == '__main__':
    p = DemoData(kind='example1', n_obs=500)
    X = p.coords[:, :2]
    y = p.coords[:, 2]

    p.plot()

    # Fit an Earth model
    model = Earth(max_terms=2)
    model.fit(X, y)
        
    # Print the model
    # print(model.summary())
        
    # Plot the model
    y_hat = model.predict(X)

    # partition the observations into two sets based on the MARS model knots
    knot = model.basis_[1].get_knot()
    var_index = model.basis_[1].get_variable()

    U = {
        'coords': np.array([item for item in p.coords if item[var_index] < knot]),
        'index': [i for i, item in enumerate(p.coords) if item[var_index] < knot]
    }
    V = {
        'coords': np.array([item for item in p.coords if item[var_index] >= knot]),
        'index': [i for i, item in enumerate(p.coords) if item[var_index] >= knot]
    }

    # run PCA on the predicted MARS values
    pca_U = PCA(n_components=2)
    pca_coords_U = pca_U.fit_transform(U['coords'])

    pca_V = PCA(n_components=2)
    pca_coords_V = pca_V.fit_transform(V['coords'])

    ax = plt.axes(projection='3d')

    ax.scatter3D(U['coords'][:, 0], U['coords'][:, 1], y_hat[U['index']], color='red')
    ax.scatter3D(V['coords'][:, 0], V['coords'][:, 1], y_hat[V['index']], color='blue')
    ax.scatter3D(X[:, 0], X[:, 1], y, color='black')
    plt.title('Simple Earth Example')
    plt.show()

    # ax = plt.axes()
    # ax.scatter(U['coords'][:, 0], U['coords'][:, 1], color='red')
    # ax.scatter(V['coords'][:, 0], V['coords'][:, 1], color='blue')
    # plt.show()