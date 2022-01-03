import numpy as np
from numpy import ndarray


def design_matrix(n_obs: int) -> np.ndarray:
    """ Create the design matrix """

    return np.random.binomial(1, p=.5, size=(n_obs, n_obs))


def covariance(design: np.ndarray, biased=False):
    """ Accepts design matrix and returns covariance matrix """
    design = design.astype(np.float64)

    mu = np.mean(design, axis=1)

    factor = len(design[:, 0]) - 1

    # center the design matrix
    design -= mu[:, None]  # this is the same as mu.reshape(-1,1)

    if biased:
        factor += 1

    return np.dot(design, design.T)*np.true_divide(1, factor)


def pca(cov_matrix: np.ndarray):
    """ Accepts a covariance matrix and returns the principal components """

    pass


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    n = 5
    x: ndarray = design_matrix(n)
    cov = covariance(x)
