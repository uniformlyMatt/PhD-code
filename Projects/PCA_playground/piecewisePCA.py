from re import A
import numpy as np
import numpy.linalg as LA
from scipy.special import erf, erfc
from sklearn.decomposition import PCA
from demodata import DemoData
from config import *

from alive_progress import alive_bar

import matplotlib.pyplot as plt
import seaborn as sns

class Problem:
    def __init__(self, n_obs, data_type='mixture', latent_dimension=2, em_tolerance=1, max_iterations=25):
        self.q = latent_dimension
        self.N = n_obs  # number of observations

        # define step size for gradient ascent
        self.step_size = 0.01

        # setup the observations
        if data_type == 'example1':
            self.data = DemoData(kind=data_type, n_obs=self.N)
        elif data_type == 'example2':
            self.data = DemoData(kind=data_type, n_obs=self.N)

        # get the coordinates from the object
        self.coords = self.data.coords

        self.p = self.coords.T.shape[0]   # dimension of the observation space
        
        # stopping criteria for optimization algorithms
        self.em_tolerance = em_tolerance   # tolerance for the loglikelihood in the EM algorithm
        self.max_iterations = max_iterations

        # initialize a set of means and standard deviations for the latent variables
        self.latent_means = [np.random.randn(self.q, 1) for _ in range(self.N)]
        self.latent_variances = [np.ones(self.q) for _ in range(self.N)]
        self.latent_Sigmas = [np.diag(var) for var in self.latent_variances]

        # set starting values for sigma2, mu1, mu2, B_1, B_2
        self.mu1 = np.ones(shape=(1, self.p))
        self.mu2 = -np.ones(shape=(1, self.p))
        self.sigma2 = np.random.rand()       # set to random positive number
        self.B1 = np.random.randn(self.p, self.q)
        self.B2 = np.random.randn(self.p, self.q)

        # indexes that correspond to the two separate subsets of the latent data
        self.I1 = [index for index, point in enumerate(self.coords) if point[0].item() >= 0]
        self.I2 = [index for index, point in enumerate(self.coords) if point[0].item() < 0]

        # I want the observations to be 1xp arrays for later computations
        self.Y = [yi.reshape(1, -1) for yi in self.coords]

        # calculate the current loglikelihood
        self.loglik = self.loglikelihood()

    def __str__(self):
        """ Overloading the print function. This presents an informative 
            string display when the object is printed.
        """
        result = "\nPiecewise Linear Probabilistic PCA\n"
        result += '='*(len(result)-2) + '\n\n'

        result += 'Model parameters\n\n'
        result += 'Number of observations: {}\nData dimensions: {}\nLatent dimensions: {}\n'.format(self.N, self.p, self.q)
        result += 'Initial Log-likelihood: {}\n'.format(self.loglik)
        result += 'Initial sigma_2: {}'.format(self.sigma2)

        return result

    def update_index(self):
        """ Update the attributes I1 and I2; these 
            indicate which latent variable are in
            Omega plus and Omega minus
        """

        self.I1 = [index for index, mi in enumerate(self.latent_means) if mi[-1] >= 0]
        self.I2 = [index for index, mi in enumerate(self.latent_means) if mi[-1] < 0]

    def loglikelihood(self):
        """ Compute the log-likelihood function.
            Compute the summation terms for the log-likelihood. 
        """
        # update the latent variances based on the updated latent covariances matrices
        self.latent_variances = [np.diag(Si) for Si in self.latent_Sigmas]

        # add up the terms that are valid over the entire index 1 <= i <= N
        determinants = [np.prod(np.diag(Sigma)) for Sigma in self.latent_Sigmas]
        traces = [np.trace(Sigma) for Sigma in self.latent_Sigmas]
        miTmi = [np.sqrt(sum(mi**2)).item() for mi in self.latent_means]

        global_terms = [0.5*(np.log(det) - tr - mi_norm) for tr, det, mi_norm in zip(traces, determinants, miTmi)]

        # now for the terms in Omega_plus
        # update the index sets
        self.update_index()

        def get_omega_terms(plus=True):
            """ Compute the summation terms over Omega plus or Omega minus """
            if plus:
                B =  self.B1    
                error_func = lambda x: SQRT_PI_OVER_2*erfc(x/ROOT2)
                exp_func = lambda x: np.exp(-0.5*x**2)
                index_set = self.I1
                mu = self.mu1

            else:
                B = self.B2
                error_func = lambda x: SQRT_PI_OVER_2*(erf(x/ROOT2) + 1)
                exp_func = lambda x: -np.exp(-0.5*x**2)
                index_set = self.I2
                mu = self.mu2

            BTB = np.matmul(B.T, B)

            # these are the 'delta_i' parameters when nu=e_q, beta=0
            deltas = [-self.latent_means[i][-1]/np.sqrt(self.latent_variances[i][-1]) for i in index_set]
            
            _BTBSigmas = [np.matmul(BTB, self.latent_Sigmas[i]) for i in index_set]

            _diagonal_terms = [BTBS[-1][-1] for BTBS in _BTBSigmas]
            _exp_terms = [delta*exp_func(delta) for delta in deltas]
            _erf_terms = [error_func(delta) for delta in deltas]
            _trace_terms = [np.trace(BTBS) for BTBS in _BTBSigmas]
            _quadratic_terms = [
                np.matmul(
                    np.matmul(
                        self.latent_means[i].T,
                        BTB
                    ),
                    self.latent_means[i]
                ) for i in index_set
            ]
            _yi_terms = [
                np.matmul(
                    self.Y[i] - mu,
                    (self.Y[i] - mu).T - 2*np.matmul(B, self.latent_means[i])
                ) for i in index_set
            ]

            _terms = [
                item1*item2.item() + item3*(item4 + item5 + item6) for item1, 
                item2, 
                item3, 
                item4, 
                item5, 
                item6 in zip(
                    _diagonal_terms,
                    _exp_terms,
                    _erf_terms,
                    _trace_terms,
                    _quadratic_terms,
                    _yi_terms
                )
            ]

            return _terms

        self.omega_plus_terms = get_omega_terms(plus=True)
        self.omega_minus_terms = get_omega_terms(plus=False)

        # finally, compute the scalars that are independent of the data and latent variables
        scalars = 0.5*self.N*(self.q - self.p*np.log(TWOPI*self.sigma2))

        # add all of the terms together
        total = 0.5*np.sum(global_terms) - (TWOPI**-0.5)/(2*self.sigma2)*(np.sum(self.omega_plus_terms) + np.sum(self.omega_minus_terms))

        return total + scalars

    def gradient_wrt_latent_Sigma(self, Si):
        """ Compute the gradient of the variational 
            lower bound with respect to a single latent
            covariance matrix over the correct subset
            of the data.
        """
        s_iq = np.sqrt(Si[-1])
        delta_i = -self.latent_means[self.index][-1]/s_iq

        if self.index in self.I1:
            B = self.B1
            erf_func = lambda x: SQRT_PI_OVER_2*erfc(x/ROOT2)
            j = 1
        elif self.index in self.I2:
            B = self.B2
            erf_func = lambda x: SQRT_PI_OVER_2*(erf(x/ROOT2) + 1)
            j = 2

        # reshape the latent covariance matrix from a flat vector to a matrix
        Si = Si.reshape(self.q, self.q)

        BTB = np.matmul(B.T, B)
        BTBS = np.matmul(BTB, Si)

        # we need the matrix containing only the last column of BTB (this is B^T*B*eq*eq^T)
        _last_col = np.zeros((self.q, self.q))
        _last_col[:, -1] = BTB[:, -1]

        # since Si is a diagonal matrix, we know the inverse is just the reciprocal
        Si_inverse = np.diag(1/np.diag(Si))
        identity = np.eye(self.q)
        _trace = np.trace(BTBS)
        
        # compute the partial derivative of delta_i wrt Sigma_i
        _partial_delta_wrt_Sigma = np.zeros((self.q, self.q))
        _partial_delta_wrt_Sigma[-1, -1] = 0.5*delta_i

        _result = 0.5*(
            Si_inverse - identity - (TWOPI**(-0.5)/self.sigma2)*(
                (-1)**(j-1)*np.exp(-delta_i**2/2)*(
                    delta_i*_last_col + _partial_delta_wrt_Sigma*(
                        BTBS[-1, -1]*(1-delta_i**2) - _trace
                    )
                ) + erf_func(delta_i)*BTB
            )
        )

        return _result.ravel()

    def gradient_wrt_latent_mean(self, mi):
        """ Computes the gradient of the loglikelihood with 
            respect to a single latent mean over Omega plus
            or Omega minus.
        """

        self.update_index()

        # these are the 's' parameters when nu=e_q, beta=0
        Sigma = self.latent_Sigmas[self.index]

        s_iq = np.sqrt(Sigma.flatten()[-1])
        delta_i = -mi[-1]/s_iq
        
        yi = self.Y[self.index]
        mi = self.latent_means[self.index]

        if self.index in self.I1:
            error_func = lambda x: erfc(x/ROOT2)
            exp_func = lambda x: np.exp(x)
            B = self.B1
            mu = self.mu1
            j = 1
        elif self.index in self.I2:
            error_func = lambda x: erf(x) + 1
            exp_func = lambda x: -np.exp(x)
            B = self.B2
            mu = self.mu2
            j = 2

        BTB = np.matmul(B.T, B)
        BTBS = np.matmul(BTB, Sigma)
        BTBmi = np.matmul(BTB, mi)
        BTymu = np.matmul(B.T, yi.T-mu)
        Bmi = np.matmul(B, mi)

        _quadratic_term = np.matmul(
            np.matmul(
                mi.T,
                BTB
            ),
            mi
        ).item()
        _yi_term = np.matmul(
            yi-mu.T,
            yi.T-mu - 2*Bmi
        ).item()
        _trace_term = np.trace(BTBS)

        _cj = (TWOPI)**(-0.5)*exp_func(delta_i)*(BTBS[-1, -1]*(1-delta_i**2) - _trace_term - _quadratic_term - _yi_term)/s_iq
        _erf_term = error_func(delta_i)*(BTBmi - BTymu)

        _eq_term = np.zeros((self.q, 1))
        _eq_term[-1] = _cj.item()

        _result = -mi + (-1)**(j-1)*(_eq_term + _erf_term)/(2*self.sigma2)

        return _result.flatten()

    def gradient_descent(self, indicator: str):
        """ Perform gradient descent until tolerance is reached. """

        assert indicator in ['means', 'covariances']

        if indicator == 'means':
            update_set = self.latent_means
            grad_func = self.gradient_wrt_latent_mean
            resize = lambda x: x.reshape(-1, 1)
        elif indicator == 'covariances':
            update_set = self.latent_Sigmas
            grad_func = self.gradient_wrt_latent_Sigma
            resize = lambda x: np.diag(np.diag(x.reshape(self.q, self.q))) # this zeros all off-diagonal elements

        # optimize latent parameters using gradient descent
        # with alive_bar(self.N) as bar:
        # update each parameter in the 'update_set'
        for index, par in enumerate(update_set):
            self.index = index

            loglik_old = self.loglikelihood()

            par_old = par.flatten()
            par_new  = par_old - self.step_size*grad_func(par_old)

            loglik_new = self.loglikelihood()
            
            count1 = 0
            # update until relative error is less than tolerance
            while np.abs((loglik_old-loglik_new)/loglik_old) > 0.01:
                par_old = par_new
                loglik_old = loglik_new.copy()

                par_new = par_old - self.step_size*grad_func(par_old)
                loglik_new = self.loglikelihood()

                count1 += 1
                if count1 > self.max_iterations:
                    print('Max iterations reached before tolerance...\n')
                    break

                # decrease the step size when we're getting close to a solution
                # if np.abs(loglik_old-loglik_new) < 2*self.tolerance:
                #     self.step_size *= .5
                
            # update the latent parameter with new optimum
            update_set[index] = resize(par_new)
                # bar()

    def M_step(self):
        """ Update the model parameters based on the maximum likelihood estimates 
            for the parameters.

            This comprises the M-step of the EM algorithm.
        """
        deltas = [-mi[-1]/np.sqrt(si[-1]) for mi, si in zip(self.latent_means, self.latent_variances)]

        # update mu_1 and mu_2
        def update_mu(plus=True):
            """ Get terms for mu_1 or mu_2 """
            if plus:
                err_func = lambda x: erfc(x/ROOT2)
                B = self.B1
                index_set = self.I1
            else:
                err_func = lambda x: erf(x/ROOT2) + 1
                B = self.B2
                index_set = self.I2

            _di = [deltas[i].item() for i in index_set]
            _mi_terms = [np.matmul(B, self.latent_means[i]) for i in index_set]
            _yi = [self.Y[i] for i in index_set]

            _reciprocal = sum([err_func(di) for di in _di])
            _sum_term = sum([err_func(di)*(yi.T - mi_term) for di, yi, mi_term in zip(_di, _yi, _mi_terms)])

            return _sum_term/_reciprocal

        self.mu1 = update_mu(plus=True)
        self.mu2 = update_mu(plus=False)
        
        # update the linear transformations B1 and B2
        def update_B(plus=True):
            """ Get the terms involving either B1 or B2, 
                depending on whether we are in Omega plus
                or Omega minus
            """
            if plus:
                error_func = lambda x: SQRT_PI_OVER_2*erfc(x/ROOT2)
                exp_func = lambda x: np.exp(-x**2/2)
                index_set = self.I1
                mu = self.mu1
            else:
                error_func = lambda x: SQRT_PI_OVER_2*(erf(x/ROOT2) + 1)
                exp_func = lambda x: -np.exp(-x**2/2)
                index_set = self.I2
                mu = self.mu2

            _not_inverted = sum([np.matmul(self.Y[i].T-mu, self.latent_means[i].T) for i in index_set])

            # compute terms for the part that gets inverted
            _si = [self.latent_Sigmas[i][-1][-1] for i in index_set]
            _mi = [self.latent_means[i][-1] for i in index_set]
            _deltas = [-mi/si for mi, si in zip(_mi, _si)]
            
            # the (qxq)th basis vector for qxq matrices
            _Mq = np.zeros((self.q, self.q))
            _Mq[-1, -1] = 1
            
            _to_invert_term1 = [di*exp_func(di)*si*_Mq for di, si in zip(_deltas, _si)]
            _to_invert_term2 = [
                error_func(di)*(
                    self.latent_Sigmas[i] + np.matmul(
                        self.latent_means[i],
                        self.latent_means[i].T
                    )
                ) for di, i in zip(_deltas, index_set)
            ]

            # compute the inverse
            _to_invert = sum([item1 + item2 for item1, item2 in zip(_to_invert_term1, _to_invert_term2)])

            _inverted_part = LA.inv(_to_invert)

            return np.matmul(_not_inverted, _inverted_part)
        
        self.B1 = update_B(plus=True)
        self.B2 = update_B(plus=False)

        # update sigma_squared
        self.sigma2 = (TWOPI**(-0.5)/(self.N*self.p))*(np.sum(self.omega_plus_terms) + np.sum(self.omega_minus_terms))

        # optimize variational parameters using gradient descent
        # print('\nOptimizing variational parameters...')
        self.gradient_descent(indicator='means')
        self.gradient_descent(indicator='covariances')

    def optimize_model(self):
        """ Perform the EM algorithm over the model parameters B1, B2, and sigma2 
            and latent means and covariance matrices
        """

        # keep track of loglikelihood
        current_loglik = self.loglikelihood()
        previous_loglik = 10*current_loglik
        count = 0

        self._loglik_results = [current_loglik]

        # keep performing EM until the change in loglikelihood is less than the tolerance
        print('Optimizing model parameters using the EM algorithm...')
        # for _ in range(20):
        
        while np.abs(current_loglik - previous_loglik) > self.em_tolerance:
            # update the model parameters
            self.M_step()
            
            previous_loglik = current_loglik
            current_loglik = self.loglikelihood()
            self._loglik_results.append(current_loglik)
            count += 1
            print(count)

            if count > self.max_iterations:
                print('Maximum iterations reached...')
                break

        self.loglik = self.loglikelihood()
        # self._loglik_results.append(self.loglik.copy())

        print('Optimal parameters reached in {} iterations.'.format(count+1))
        print('Log-likelihood at the optimum: {}'.format(self.loglik))
        
    def get_result(self):
        """ Estimate the positions of the latent variables """

        self.W1 = [self.latent_means[index] for index in self.I1]
        self.W2 = [self.latent_means[index] for index in self.I2]

        self.result_x1 = [row[0] for row in self.W1]
        self.result_y1 = [row[1] for row in self.W1]
        self.result_z1 = [0]*len(self.W1)

        self.result_x2 = [row[0] for row in self.W2]
        self.result_y2 = [row[1] for row in self.W2]
        self.result_z2 = [0]*len(self.W2)

def plot_example1(problem, pca_coords=None):
    """ Plot the results of the piecewise PCA and compares to
        regular PCA. 
        
        Example 1

        Note: this is only used when latent_dim = 2.

        :args:
        ------
        problem: 
            instance of the Problem class

        pca_coords:
            results from classical PCA with the observations
    """

    # reshape the latent means for plotting
    m = np.vstack([row.reshape(1, -1) for row in problem.latent_means])

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(6, 15))

    # plot the observations
    obs1 = problem.coords[problem.I1]
    obs2 = problem.coords[problem.I2]
    ax1.scatter(obs1[:, 0], obs1[:, 1], color='#d82f49ff')
    ax1.scatter(obs2[:, 0], obs2[:, 1], color='#6a2357ff')
    # problem.data.plot(ax=ax1)
    ax1.set_title('Observations')

    # plot the latent space - piecewise PCA
    m1 = m[problem.I1]
    m2 = m[problem.I2]
    ax2.scatter(m1[:, 0], m1[:, 1], color='#d82f49ff')
    ax2.scatter(m2[:, 0], m2[:, 1], color='#6a2357ff')
    ax2.set_aspect('equal')
    ax2.set_title('Estimated positions in latent space - piecewise PCA')

    # plot the latent space - classical PCA
    if pca_coords is not None:
        x1 = [item[0] for item in pca_coords[problem.I1]]
        y1 = [item[1] for item in pca_coords[problem.I1]]
        x2 = [item[0] for item in pca_coords[problem.I2]]
        y2 = [item[1] for item in pca_coords[problem.I2]]

        ax3.scatter(x1, y1, color='#d82f49ff')
        ax3.scatter(x2, y2, color='#6a2357ff')
        ax3.set_aspect('equal')
        ax3.set_title('Estimated positions in latent space - PCA')

    plt.show()
    fig.savefig('results/example1_result.png', bbox_inches='tight')

def plot_example2(problem, pca_coords=None):
    """ Plot the results of the piecewise PCA and compares to
        regular PCA. 
        
        Example 2

        Note: this is only used when latent_dim = 2.

        :args:
        ------
        problem: 
            instance of the Problem class

        pca_coords:
            results from classical PCA with the observations
    """
    fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(6, 12))

    obs1 = problem.coords[problem.I1]
    obs2 = problem.coords[problem.I2]

    x1 = obs1[:, 0]
    y1 = obs1[:, 1]
    z1 = obs1[:, 2]

    x2 = obs2[:, 0]
    y2 = obs2[:, 1]
    z2 = obs2[:, 2]

    # plot the observations
    fig2 = plt.figure(figsize=(6, 6))
    ax3 = fig2.add_subplot(projection='3d')
    ax3.scatter3D(x1, y1, z1, color='#d82f49ff')
    ax3.scatter3D(x2, y2, z2, color='#6a2357ff')
    ax3.azim = -80
    ax3.elev = 16
    ax3.set_title('Observations')
    fig2.savefig('results/example2_observations.png', bbox_inches='tight')

    # plot the latent space - piecewise PCA
    # reshape the latent means for plotting
    m = np.vstack([row.reshape(1, -1) for row in problem.latent_means])
    lat1 = m[problem.I1]
    lat2 = m[problem.I2]
    ax1.scatter(lat1[:, 0], lat1[:, 1], color='#d82f49ff')
    ax1.scatter(lat2[:, 0], lat2[:, 1], color='#6a2357ff')
    ax1.set_aspect('equal')
    ax1.set_title('Estimated positions in latent space - piecewise PCA')

    # plot the latent space - classical PCA
    if pca_coords is not None:
        x1 = [item[0] for item in pca_coords[problem.I1]]
        y1 = [item[1] for item in pca_coords[problem.I1]]
        x2 = [item[0] for item in pca_coords[problem.I2]]
        y2 = [item[1] for item in pca_coords[problem.I2]]

        ax2.scatter(x1, y1, color='#d82f49ff')
        ax2.scatter(x2, y2, color='#6a2357ff')
        ax2.set_aspect('equal')
        ax2.set_title('Estimated positions in latent space - PCA')

    plt.show()
    fig1.savefig("results/example2_result.png", bbox_inches='tight')

def distance_to_mean(problem):
    """ Calculate the distance between the data points and their mean 
        for both the observations and the points in latent space.

        I want to see if the distances to the mean were conserved.
    """
    obs_mean = np.mean(problem.coords, axis=0)
    result_m = np.vstack([row.reshape(1, -1) for row in problem.latent_means])
    result_mean = np.mean(result_m, axis=0)

    # compute the distances between points and their mean
    obs_dist = [np.round(LA.norm(obs - obs_mean, 2)**2, 2) for obs in problem.coords]
    result_dist = [np.round(LA.norm((point - result_mean)**2, 2)) for point in result_m]

    ax = plt.axes()
    ax.scatter(obs_dist, result_dist, color='black', marker='o')
    ax.set_xlabel('Observation distance to empirical mean')
    ax.set_ylabel('Latent distance to latent mean')
    # ax.plot(x, result_dist, color='blue')
    plt.show()

def example(n_obs, data_type, latent_dimension=2):
    """ Compute Example 1 or 2 """

    p = Problem(n_obs=n_obs, data_type=data_type, latent_dimension=latent_dimension, em_tolerance=1, max_iterations=10)
    print(p)

    p.optimize_model()
    print('\nPosterior model mean parameters...')
    print(p.mu1, p.mu2)
    print('\nPosterior transformations...')
    print(p.B1, p.B2)

    # with open('{} - index sets.txt'.format(data_type), 'w') as file:
    #     file.write(str(p.I1))
    #     file.write(str(p.I2))

    # for comparison, let's do classical PCA with the data
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(p.coords)

    sns.set_theme()
    # distance_to_mean(p)

    # now plot piecewise pca and classical PCA
    if data_type == 'example1':
        plot_example1(p, pca_coords=pca_coords)
    elif data_type == 'example2':
        plot_example2(p, pca_coords=pca_coords)

if __name__ == '__main__':
    # Example 1
    # example(100, 'example1')

    # Example 2
    example(100, 'example2')