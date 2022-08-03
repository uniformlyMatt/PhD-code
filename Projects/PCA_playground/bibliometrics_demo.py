import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from piecewisePCA import Problem
from pyearth import Earth

def plot_result(df, model, pca_coords):
    """ Plot the results of the piecewise PCA and compares to
        regular PCA. 
        
        bibliometrics

        Note: this is only used when latent_dim = 2.

        :args:
        ------
        model: 
            instance of the Problem class

        pca_coords:
            results from classical PCA with the observations
    """
    fig1, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    obs1 = df[['log_publications', 'log_journal_h_index_mean', 'log_SJR_mean']].loc[model.I1]
    obs2 = df[['log_publications', 'log_journal_h_index_mean', 'log_SJR_mean']].loc[model.I2]

    x1 = obs1['log_publications']
    y1 = obs1['log_journal_h_index_mean']
    z1 = obs1['log_SJR_mean']

    x2 = obs2['log_publications']
    y2 = obs2['log_journal_h_index_mean']
    z2 = obs2['log_SJR_mean']

    # plot the observations
    fig2 = plt.figure(figsize=(6, 6))
    ax3 = fig2.add_subplot(projection='3d')
    ax3.scatter3D(x1, y1, z1, color='#d82f49ff')
    ax3.scatter3D(x2, y2, z2, color='#6a2357ff')
    # ax3.scatter3D(x1[2], y1[2], z1[2], color='blue', marker='X', s=100)
    ax3.set_xlabel('log-publications')
    ax3.set_ylabel('log-mean journal h-index')
    ax3.set_zlabel('log-mean SJR')
    ax3.azim = -177
    ax3.elev = 34
    ax3.set_title('Bibliometrics - Observations')

    # plot the latent space - piecewise PCA
    # reshape the latent means for plotting
    m = np.vstack([row.reshape(1, -1) for row in model.latent_means])
    lat1 = m[model.I1]
    lat2 = m[model.I2]
    ax1.scatter(lat1[:, 0], lat1[:, 1], color='#d82f49ff')
    ax1.scatter(lat2[:, 0], lat2[:, 1], color='#6a2357ff')
    # ax1.scatter(lat1[2, 0], lat1[2, 1], color='blue', marker='X', s=100)
    ax1.set_aspect('equal')
    ax1.set_title('Estimated positions in latent space - piecewise PCA')

    # plot the latent space - classical PCA
    if pca_coords is not None:
        x1 = [item[0] for item in pca_coords[model.I1]]
        y1 = [item[1] for item in pca_coords[model.I1]]
        x2 = [item[0] for item in pca_coords[model.I2]]
        y2 = [item[1] for item in pca_coords[model.I2]]

        ax2.scatter(x1, y1, color='#d82f49ff')
        ax2.scatter(x2, y2, color='#6a2357ff')
        # ax2.scatter(x1[2], y1[2], color='blue', marker='X', s=100)
        ax2.set_aspect('equal')
        ax2.set_title('Estimated positions in latent space - PCA')

    plt.show()

if __name__ == '__main__':
    df = pd.read_excel('pca_inputs_standardized.xlsx')
    df.info()
    latent_dim = 2

    # plot pairwise scatterplots for all variables
    sns.pairplot(df, corner=True, markers='.')
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

    # classical PCA model
    pca = PCA(n_components=latent_dim)

    # piecewise PCA model
    model = Problem(n_obs=N, data_type='real', latent_dimension=latent_dim, input_data={'X': X, 'y': y})

    # fit the piecewise PCA model
    model.optimize_model()
    
    # get the latent space representations for piecewise and classical PCA
    obs1 = np.vstack([model.Y[i] for i in model.I1])
    obs2 = np.vstack([model.Y[i] for i in model.I2])

    latent_points = np.vstack((obs1, obs2))
    latent_pca = PCA(n_components=latent_dim)
    latent_pca_coords = latent_pca.fit_transform(latent_points) # orthogonalize the piecewise PCA results

    pca_coords = pca.fit_transform(X=df)

    explained_rho2 = model.rho_squared()*latent_pca.explained_variance_ratio_
    pca_var = pca.explained_variance_ratio_
    
    print("BIC: {}".format(model.BIC()))
    print('Total explained variance:')
    print('Piecewise PCA: {}\nClassical PCA: {}'.format(sum(explained_rho2), sum(pca_var)))
    print('Explained variance:')
    print('Piecewise PCA\n Dim 1: {}\n Dim 2: {}'.format(explained_rho2[0], explained_rho2[1]))
    print('Classical PCA\n Dim 1: {}\n Dim 2: {}'.format(pca_var[0], pca_var[1]))
    print('Pseudo-R2: {}'.format(model.rho_squared()))

    sns.set_theme()
    # now plot piecewise pca and classical PCA
    plot_result(df=df, model=model, pca_coords=pca_coords)