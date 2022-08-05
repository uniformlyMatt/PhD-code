import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from piecewisePCA import Problem
from pyearth import Earth

def pickle_model(model):
    """ Pickles a PWPCA model """

    B1 = model.B1
    B2 = model.B2
    latent_means = model.latent_means
    I1 = model.I1
    I2 = model.I2
    rho2 = model.rho_squared()
    BIC = model.BIC()

    # get the latent space representations for piecewise and classical PCA
    m1 = np.vstack([latent_means[i].reshape(1, -1) for i in I1])
    m2 = np.vstack([latent_means[i].reshape(1, -1) for i in I2])

    try:
        with open('model_B1.pkl', 'wb') as file:
            pickle.dump(B1, file, protocol=pickle.HIGHEST_PROTOCOL)
        with open('model_B2.pkl', 'wb') as file:
            pickle.dump(B2, file, protocol=pickle.HIGHEST_PROTOCOL)
        with open('model_I1.pkl', 'wb') as file:
            pickle.dump(I1, file, protocol=pickle.HIGHEST_PROTOCOL)
        with open('model_I2.pkl', 'wb') as file:
            pickle.dump(I2, file, protocol=pickle.HIGHEST_PROTOCOL)
        with open('model_rho2.pkl', 'wb') as file:
            pickle.dump(rho2, file, protocol=pickle.HIGHEST_PROTOCOL)
        with open('model_BIC.pkl', 'wb') as file:
            pickle.dump(BIC, file, protocol=pickle.HIGHEST_PROTOCOL)
        with open('model_m1.pkl', 'wb') as file:
            pickle.dump(m1, file, protocol=pickle.HIGHEST_PROTOCOL)
        with open('model_m2.pkl', 'wb') as file:
            pickle.dump(m2, file, protocol=pickle.HIGHEST_PROTOCOL)
    except:
        raise

    return 0

if __name__ == '__main__':
    df = pd.read_excel('pca_inputs_standardized.xlsx')
    df.info()
    latent_dim = 2

    # plot pairwise scatterplots for all variables
    g = sns.pairplot(df, corner=True, markers='.', height=2, aspect=1.)
    for ax in g.axes.flatten():
        if ax:
            ax.set_xlabel(ax.get_xlabel(), rotation=-90)
            ax.set_ylabel(ax.get_ylabel(), rotation=0)
            ax.yaxis.get_label().set_horizontalalignment('right')
            
    plt.show()

    # show the piecewise linear relationship between log-mean journal h-index and log-mean SJR
    x_demo = df['log_journal_h_index_mean'].values.reshape(-1, 1)
    y_demo = df['log_SJR_mean'].values.reshape(-1, 1)

    model_demo = Earth(max_terms=2)
    model_demo.fit(x_demo, y_demo)
    xhat = np.linspace(x_demo.min(), x_demo.max(), 101).reshape(-1, 1)
    yhat = model_demo.predict(xhat)

    fig, ax = plt.subplots()
    ax.scatter(x_demo, y_demo, marker='.')
    ax.plot(xhat, yhat, color='black')
    ax.set_xlabel('log-mean journal h-index')
    ax.set_ylabel('log-mean SJR')
    plt.show()

    # prepare data for piecewise PCA
    X = df.drop('log_SJR_mean', axis=1).values
    y = df['log_SJR_mean'].values.reshape(-1, 1)

    N = len(df)

    # piecewise PCA model
    model = Problem(n_obs=N, data_type='real', latent_dimension=latent_dim, input_data={'X': X, 'y': y})

    # fit the piecewise PCA model
    model.optimize_model()

    pickle_model(model)
    
    # latent_points = np.vstack((means_I1, means_I2))
    # latent_pca = PCA(n_components=latent_dim)
    # latent_pca_coords = latent_pca.fit_transform(latent_points) # PCA to compute the explained variance

    # pca_coords = pca.fit_transform(X=df)

    # with open('bibliometrics_rho2.txt', 'r') as file:
    #     _rho2 = float(file.read())
        
    # # _rho2 = model.rho_squared()
    # explained_rho2 = _rho2*latent_pca.explained_variance_ratio_
    # pca_var = pca.explained_variance_ratio_
    
    # # print("\nBIC: {}".format(model.BIC()))
    # print('Total explained variance:')
    # print('PWPCA: {}\nPCA: {}'.format(sum(explained_rho2), sum(pca_var)))
    # print('Explained variance by dimension:')
    # print('PWPCA\n Dim 1: {}\n Dim 2: {}'.format(explained_rho2[0], explained_rho2[1]))
    # print('PCA\n Dim 1: {}\n Dim 2: {}'.format(pca_var[0], pca_var[1]))
