import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def plot_result(df, I1, I2, pwpca_coords, pca_coords):
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

    obs1 = df[['log_publications', 'log_journal_h_index_mean', 'log_SJR_mean']].loc[I1]
    obs2 = df[['log_publications', 'log_journal_h_index_mean', 'log_SJR_mean']].loc[I2]

    x1 = obs1['log_publications']
    y1 = obs1['log_journal_h_index_mean']
    z1 = obs1['log_SJR_mean']

    x2 = obs2['log_publications']
    y2 = obs2['log_journal_h_index_mean']
    z2 = obs2['log_SJR_mean']

    # plot the observations
    fig2 = plt.figure(figsize=(6, 6))
    ax3 = fig2.add_subplot(projection='3d')
    ax3.scatter3D(x1, y1, z1, color='#d82f49ff', alpha=0.5)
    ax3.scatter3D(x2, y2, z2, color='#6a2357ff', alpha=0.5)
    # ax3.scatter3D(x1[2], y1[2], z1[2], color='blue', marker='X', s=100)
    ax3.set_xlabel('log-publications')
    ax3.set_ylabel('log-mean journal h-index')
    ax3.set_zlabel('log-mean SJR')
    ax3.azim = -177
    ax3.elev = 34
    ax3.set_title('Bibliometrics - Observations')

    # plot the latent space - piecewise PCA
    # reshape the latent means for plotting
    # m = np.vstack([row.reshape(1, -1) for row in model.latent_means])
    lat1 = pwpca_coords[0]
    lat2 = pwpca_coords[1]
    ax1.scatter(lat1[:, 0], lat1[:, 1], color='#d82f49ff', alpha=0.5)
    ax1.scatter(lat2[:, 0], lat2[:, 1], color='#6a2357ff', alpha=0.5)
    # ax1.scatter(lat1[2, 0], lat1[2, 1], color='blue', marker='X', s=100)
    ax1.set_aspect('equal')
    ax1.set_title('Estimated positions in latent space - piecewise PCA')

    # plot the latent space - classical PCA
    if pca_coords is not None:
        x1 = [item[0] for item in pca_coords[I1]]
        y1 = [item[1] for item in pca_coords[I1]]
        x2 = [item[0] for item in pca_coords[I2]]
        y2 = [item[1] for item in pca_coords[I2]]

        ax2.scatter(x1, y1, color='#d82f49ff', alpha=0.5)
        ax2.scatter(x2, y2, color='#6a2357ff', alpha=0.5)
        # ax2.scatter(x1[2], y1[2], color='blue', marker='X', s=100)
        ax2.set_aspect('equal')
        ax2.set_title('Estimated positions in latent space - PCA')

    plt.show()

if __name__ == '__main__':
    df = pd.read_excel('pca_inputs_standardized.xlsx')
    df.info()
    latent_dim = 2

    # classical PCA model
    pca = PCA(n_components=latent_dim)
    
    # load the pickled PWPCA model parameters and index sets
    with open('model_I1.pkl', 'rb') as file:
        I1 = pickle.load(file)
    with open('model_I2.pkl', 'rb') as file:
        I2 = pickle.load(file)
    with open('model_B1.pkl', 'rb') as file:
        B1 = pickle.load(file)
    with open('model_B2.pkl', 'rb') as file:
        B2 = pickle.load(file)
    with open('model_m1.pkl', 'rb') as file:
        m1 = pickle.load(file)
    with open('model_m2.pkl', 'rb') as file:
        m2 = pickle.load(file)
    with open('model_BIC.pkl', 'rb') as file:
        BIC = pickle.load(file)
    with open('model_rho2.pkl', 'rb') as file:
        rho2 = pickle.load(file)
    
    P1 = np.matmul(m1, B1.T)
    P2 = np.matmul(m2, B2.T)
    P = np.vstack((P1, P2))

    partition = P1.shape[0]
    
    latent_pca = PCA(n_components=latent_dim).fit(P)
    pca2 = PCA(n_components=latent_dim).fit(np.vstack((m1, m2)))
    print(rho2*sum(pca2.explained_variance_ratio_))

    BasisV = latent_pca.components_.T

    BasisVTBasisV = np.matmul(BasisV.T, BasisV)
    BVTBV_inv = np.linalg.inv(BasisVTBasisV)
    MV = np.matmul(P, np.matmul(BasisV, BVTBV_inv))

    MV1 = MV[:partition, :]
    MV2 = MV[partition:, :]

    pca_coords = pca.fit_transform(X=df)
        
    explained_rho2 = rho2*latent_pca.explained_variance_ratio_
    pca_var = pca.explained_variance_ratio_
    
    print('Pseudo-R2: {}'.format(rho2))
    print('\nTotal explained variance:')
    print('PWPCA: {}\nPCA: {}'.format(sum(explained_rho2), sum(pca_var)))
    print('\nExplained variance by dimension:')
    print('PWPCA\n Dim 1: {}\n Dim 2: {}'.format(explained_rho2[0], explained_rho2[1]))
    print('PCA\n Dim 1: {}\n Dim 2: {}'.format(pca_var[0], pca_var[1]))

    df_save = pd.DataFrame(np.vstack((MV1, MV2)), columns=['PWPCA_PC1', 'PWPCA_PC2'])
    df_save.info()
    df_save.to_csv('bib_decorrelated_latent_space.csv', index=False)

    sns.set_theme()
    # now plot piecewise pca and classical PCA
    # plot_result(df=df, I1=I1, I2=I2, pwpca_coords=(MV1, MV2), pca_coords=pca_coords)