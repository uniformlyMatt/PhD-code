from operator import index
import numpy as np
import numpy.linalg as LA
from scipy.special import erf, erfc
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from pyearth import Earth
from demodata import DemoData
from config import *

import matplotlib.pyplot as plt
import seaborn as sns

def _summand(yi, mi, Si, di, muj, Bj, error_func, j):
    """ Compute the summand
        (-1)**j*di*exp(-di^2/2)*[BjTBjSi]_q + error_func*_quadratic_terms
    """
    BTB = np.matmul(Bj.T, Bj)

    _quad_terms = quadratic_terms(Si=Si, yi=yi, mi=mi, muj=muj, Bj=Bj)
    _BTBS = np.matmul(BTB, Si)[-1, -1]

    _result = (-1)**j*di*np.exp(-di**2/2)*_BTBS + SQRT_PI_OVER_2*error_func(di)*_quad_terms
    return _result

def quadratic_terms(Si, yi, mi, muj, Bj):
    """ Compute Tr(B_j^TB_j) + mi^TBj^TBjmi + (yi-muj)^T(yi-muj) - 2(yi-muj)^TBjmi """
    mi = mi.reshape(-1, 1) # no effect if mi is already qx1

    BjTBj = np.matmul(Bj.T, Bj)
    Bjmi = np.matmul(Bj, mi)

    _trace_term = np.trace(
        np.matmul(
            BjTBj,
            Si
        )
    )

    _quad_term_mi = np.matmul(Bjmi.T, Bjmi)

    _yi_terms = np.matmul(
        (yi-muj).T,
        yi-muj
    ) - 2*np.matmul(
        (yi-muj).T,
        Bjmi
    )
    _result = _trace_term + _quad_term_mi + _yi_terms
    return _result

def get_omega_terms(j: int, index_set: list, deltas: list, problem):
    """ Compute the summation terms over Omega plus or Omega minus. 
        Equivalently, compute the sums over I1 or I2.
    """
    assert j in [0, 1]

    mu = problem.model_maps[j]
    B = problem.model_maps[j]
    error_func = problem.error_funcs[j]
    latent_Sigmas = problem.latent_Sigmas.copy()
    Y = problem.Y.copy()
    latent_means = problem.latent_means.copy()

    _sums = [
        _summand(
            Y[i], 
            latent_means[i], 
            latent_Sigmas[i],
            deltas[i],
            mu,
            B,
            error_func,
            j
        ) for i in index_set
    ]

    _result = np.sum(_sums)

    return _result

def obj(x, problem, index=-1, j=-1, kind=''):
    """ Accepts a single parameter 'x', an instance of the Problem class, 
        and a 'kind' parameter. Computes the log-likelihood of the VLB
        given the single model or variational parameter 'x'.

        args:
        -----
        x - np.array or float
            singled out model or variational parameter

        problem - Problem
            contains all other model and variational parameters

        kind - str
            specifies which input parameter type 'x' is
    """

    def separate_mi(mi, index, latent_means):
        """ Compute the mi-specific terms in the log-likelihood """
        latent_means.pop(index)

        _miTmi = [np.sqrt(sum(m**2)).item() for m in latent_means]
        _miTmi += [np.sqrt(sum(mi**2))]
        _result = sum(_miTmi)
        return _result

    latent_means = problem.latent_means.copy()
    latent_Sigmas = problem.latent_Sigmas.copy()

    if kind == 'mi':
        assert index >= 0
        assert j >= 0

        index_set = problem.index_sets[j].copy()
        index_set.remove(index)
        miTmi_terms = separate_mi(x, index, latent_means.copy())
        deltas = problem.deltas.copy()
        latent_Sigmas = problem.latent_Sigmas.copy()

        # get the inner terms corresponding to the index set containing mi - without including mi
        _Ij_terms = get_omega_terms(j=j, index_set=index_set, deltas=deltas, problem=problem)
        _Ij_terms += _summand(problem.Y[index], x, latent_Sigmas[index], deltas[index], problem.model_means[j], problem.model_maps[j], problem.error_funcs[j], j)

        # get the inner terms corresponding to the other index set
        k = (j+1) % 2
        _Ik_terms = get_omega_terms(j=k, index_set=problem.index_sets[k].copy(), deltas=deltas, problem=problem)

    elif kind == 'nu':
        x = x.reshape(-1, 1)

        latent_Sigma_inverses = [np.diag(1/np.diag(var)) for var in latent_Sigmas]
        plane_denoms = [np.matmul(np.matmul(x.T, Si_inv), x).item()**(-.5) for Si_inv in latent_Sigma_inverses]
        deltas = [-(np.matmul(x.T, mi).item() + problem.beta)*vi for mi, vi in zip(latent_means, plane_denoms)]
        miTmi_terms = sum([np.sqrt(sum(mi**2)).item() for mi in latent_means])

        _Ij_terms = get_omega_terms(j=0, index_set=problem.index_sets[0], deltas=deltas, problem=problem)
        _Ik_terms = get_omega_terms(j=1, index_set=problem.index_sets[1], deltas=deltas, problem=problem)

    elif kind == 'beta':
        nu = problem.nu.copy().reshape(-1, 1)

        latent_Sigma_inverses = [np.diag(1/np.diag(var)) for var in latent_Sigmas]
        plane_denoms = [np.matmul(np.matmul(nu.T, Si_inv), nu).item()**(-.5) for Si_inv in latent_Sigma_inverses]
        deltas = [-(np.matmul(nu.T, mi).item() + x)*vi for mi, vi in zip(latent_means, plane_denoms)]
        miTmi_terms = sum([np.sqrt(sum(mi**2)).item() for mi in latent_means])

        _Ij_terms = get_omega_terms(j=0, index_set=problem.index_sets[0], deltas=deltas, problem=problem)
        _Ik_terms = get_omega_terms(j=1, index_set=problem.index_sets[1], deltas=deltas, problem=problem)
    
    else:
        deltas = problem.deltas
        miTmi_terms = sum([np.sqrt(sum(mi**2)).item() for mi in latent_means])
        _Ij_terms = get_omega_terms(j=0, index_set=problem.index_sets[0], deltas=deltas, problem=problem)
        _Ik_terms = get_omega_terms(j=1, index_set=problem.index_sets[1], deltas=deltas, problem=problem)

    # add up the terms that are valid over the entire index 1 <= i <= N
    determinants = [np.prod(np.diag(Sigma)) for Sigma in latent_Sigmas]
    traces = [np.trace(Sigma) for Sigma in latent_Sigmas]
    _global_terms = 0.5*(sum([np.log(det) - tr for tr, det in zip(traces, determinants)]) - miTmi_terms)

    # finally, compute the scalars that are independent of the data and latent variables
    _scalars = 0.5*problem.N*(problem.q - problem.p*np.log(TWOPI*problem.sigma2))

    # add all of the terms together
    _total = _scalars + _global_terms + problem._const*(_Ij_terms + _Ik_terms)
    return -1*_total.item()
    
class Problem:
    def __init__(self, n_obs, data_type='example2', latent_dimension=2, em_tolerance=1e-3, max_iterations=25):
        self.q = latent_dimension
        self.N = n_obs  # number of observations

        # define step size for gradient ascent
        self.step_size = 0.01

        # setup the observations
        if data_type == 'example2':
            self.data = DemoData(kind=data_type, n_obs=self.N)

        # get the coordinates from the object
        self.Y = self.data.coords

        # initialize the latent means using a MARS model
        X = self.data.X
        y = self.data.y

        # Fit an Earth model
        model = Earth(max_terms=2)
        model.fit(X, y)
        # y_hat = model.predict(X).reshape(-1, 1)
            
        # partition the observations into two sets based on the MARS model knots
        knot = model.basis_[1].get_knot()
        var_index = model.basis_[1].get_variable()

        # this is the partitioning hyperplane in the latent space
        # self.nu = np.random.randn(self.q, 1)
        self.nu = np.zeros(shape=(self.q))
        self.nu[var_index] = knot

        self.nu = self.nu.reshape(-1, 1)
        # normalize to make nu a unit vector
        # self.nu /= LA.norm(self.nu, 2)
        self.beta = 0.

        U_index = [i for i, item in enumerate(X) if item[var_index] < knot]
        V_index = [i for i, item in enumerate(X) if item[var_index] >= knot]

        U = {
            'coords': np.array([item for item in self.Y if item[var_index] < knot]),
            'index': [i for i, item in enumerate(self.Y) if item[var_index] < knot]
        }
        V = {
            'coords': np.array([item for item in self.Y if item[var_index] >= knot]),
            'index': [i for i, item in enumerate(self.Y) if item[var_index] >= knot]
        }

        # initialize a set of means and standard deviations for the latent variables
        self.latent_means = [0]*self.N

        for i, item in zip(U['index'], U['coords']):
            self.latent_means[i] = item[:2].reshape(-1, 1)
        for i, item in zip(V['index'], V['coords']):
            self.latent_means[i] = item[:2].reshape(-1, 1)

        # self.latent_means = [np.random.randn(self.q, 1) for _ in range(self.N)]
        # self.latent_means = self.Y.copy()
        # self.latent_means = [i.reshape(-1, 1) for i in self.latent_means]
        self.latent_variances = [np.ones(self.q) for _ in range(self.N)]
        self.latent_Sigmas = [np.diag(var) for var in self.latent_variances]
        self.latent_Sigma_inverses = [np.diag(1/np.diag(var)) for var in self.latent_Sigmas]

        # compute the delta_i parameters - these are both lists of floats
        self.plane_denoms = [np.matmul(np.matmul(self.nu.T, Si_inv), self.nu).item()**(-.5) for Si_inv in self.latent_Sigma_inverses]
        self.deltas = [-(np.matmul(self.nu.T, mi).item() + self.beta)*vi for mi, vi in zip(self.latent_means, self.plane_denoms)]

        # indexes that correspond to the two separate subsets of the latent data
        self.I1 = U_index
        self.I2 = V_index

        # ax = plt.axes()
        # ax.scatter(U['coords'][:, 0], U['coords'][:, 1], color='red')
        # ax.scatter(V['coords'][:, 0], V['coords'][:, 1], color='blue')
        # plt.show()

        self.p = self.Y.T.shape[0]   # dimension of the observation space
        
        # stopping criteria for optimization algorithms
        self.em_tolerance = em_tolerance   # tolerance for the loglikelihood in the EM algorithm
        self.max_iterations = max_iterations

        ###############################
        # Initialize model parameters #
        ###############################
        # set starting values for sigma2, mu1, mu2, B_1, B_2
        self.mu1 = np.zeros((self.p, 1)) #np.mean(self.Y[self.I1], axis=0).reshape(-1, 1)
        self.mu2 = np.zeros((self.p, 1)) #np.mean(self.Y[self.I2], axis=0).reshape(-1, 1)

        self.sigma2 = 1.
        # self.B1 = np.zeros(shape=(self.p, self.q))
        # self.B2 = np.zeros(shape=(self.p, self.q))
        self.B1 = np.random.randn(self.p, self.q)
        self.B2 = np.random.randn(self.p, self.q)

        # for convenience in computing derivatives later
        self.index_sets = {0: self.I1, 1: self.I2}
        self.model_means = {0: self.mu1, 1: self.mu2}
        self.model_maps = {0: self.B1, 1: self.B2}
        self.error_funcs = {0: lambda x: erfc(x/ROOT2), 1: lambda x: erf(x/ROOT2) + 1}
        self._loglik_results = {0: [], 1: []}

        self._const = -1/(ROOT2PI*2*self.sigma2)

        # I want the observations to be px1 arrays for later computations
        self.Y = [yi.reshape(-1, 1) for yi in self.Y]

    def __str__(self):
        """ Overloading the print function. This presents an informative 
            string display when the object is printed.
        """
        result = "\nPiecewise Linear Probabilistic PCA\n"
        result += '='*(len(result)-2) + '\n\n'

        result += 'Model parameters\n\n'
        result += 'Number of observations: {}\nData dimensions: {}\nLatent dimensions: {}\n'.format(self.N, self.p, self.q)
        result += 'Initial Log-likelihood: {}\n'.format(obj(0, self))
        result += 'Initial sigma_2: {}'.format(self.sigma2)

        return result

    def update_index(self):
        """ Update the attributes I1 or I2; these 
            indicate which latent variable are in
            Omega plus and Omega minus
        """
        self.latent_Sigma_inverses = [np.diag(1/np.diag(var)) for var in self.latent_Sigmas]

        # compute the delta_i parameters - these are both lists of floats
        self.plane_denoms = [np.matmul(np.matmul(self.nu.T, Si_inv), self.nu).item()**(-.5) for Si_inv in self.latent_Sigma_inverses]
        self.deltas = [-(np.matmul(self.nu.T, mi).item() + self.beta)*vi for mi, vi in zip(self.latent_means, self.plane_denoms)]

        self.I1 = [index for index, di in enumerate(self.deltas) if di >= 0]
        self.I2 = [index for index, di in enumerate(self.deltas) if di < 0]

        self.index_sets = {0: self.I1, 1: self.I2}

        return 0

    def variational_summand(self, index, parameter, j):
        """ Computes the inner terms for the derivatives 
            dL/dmi, dL/dSigma_i, dL/dnu, and dL/dbeta.
        """
        assert j in [1, 2]
        assert parameter in ['Si', 'nu', 'beta']

        muj = self.model_means[j]
        Bj = self.model_maps[j]
        
        di = self.deltas[index]
        Si = self.latent_Sigmas[index]
        Si_inv = self.latent_Sigma_inverses[index]
        yi = self.Y[index]
        mi = self.latent_means[index]
        Di = self.plane_denoms[index]

        BjTBj = np.matmul(
            Bj.T,
            Bj
        )

        _plane_const = (np.matmul(self.nu.T, mi).item() + self.beta)

        _derivs = {
            'mi': -self.nu*Di,
            'Si': -.5*Di**3*_plane_const*np.matmul(
                Si_inv,
                np.matmul(
                    self.nu,
                    np.matmul(
                        self.nu.T,
                        Si_inv
                    )
                )
            ),
            'nu': -mi*Di + Di**3*_plane_const*np.matmul(Si_inv, self.nu),
            'beta': -Di
        }
        derivative = _derivs[parameter]

        _qq_term = np.matmul(
            BjTBj,
            Si
        )[-1, -1]

        _exp_term = np.exp(-di**2/2)
        _delta_term = (1-di**2)*_qq_term
        _quad_terms = quadratic_terms(Si, yi, mi, muj, Bj)

        return derivative*_exp_term*((-1)**(j+1)*(_delta_term - _quad_terms))

    def variational_gradient(self, index, parameter):
        """ Compute the gradient of the log-likelihood with respect 
            to a single variational parameter.

            args:
            -----
            index:  int
                index of the variational parameter, 1 <= index <= N
            parameter:  str
                valid values are either 'mi' or 'Si
        """
        assert parameter in ['mi', 'Si']
        assert index <= self.N

        if index in self.I1:
            j = 1
        elif index in self.I2:
            j = 2

        mu = self.model_means[j]
        B = self.model_maps[j]
        _error_func = self.error_funcs[j]

        BTB = np.matmul(B.T, B)
        di = self.deltas[index]

        Si_inv = self.latent_Sigma_inverses[index]
        M_qq = np.zeros(shape=(self.q, self.q))
        M_qq[:, -1] = BTB[:, -1]
        _outer_terms = .5*(Si_inv - np.eye(self.q))
        _inner_terms = di*np.exp(-di**2/2)*M_qq + SQRT_PI_OVER_2*_error_func(di)*BTB
        _inner_terms += self.variational_summand(index=index, parameter=parameter, j=j)

        _result = _outer_terms + self._const*_inner_terms

        return _result.ravel()

    def partition_gradient(self, parameter):
        """ Compute the gradient with respect to a partition 
            parameter.

            args:
            -----
            parameter: str
                either 'nu' or 'beta'
        """
        _I1_sum = sum([self.variational_summand(index=index, parameter=parameter, j=0) for index in self.I1])
        _I2_sum = sum([self.variational_summand(index=index, parameter=parameter, j=1) for index in self.I2])

        _result = self._const*(_I1_sum + _I2_sum)
        return _result

    def gradient_descent(self, indicator: str):
        """ Perform gradient descent until tolerance is reached. """
        # TODO: make this more general with less conditionals i.e. pass in a parameter, grad_func, and tolerance

        assert indicator in ['latent_covariances', 'nu', 'beta']

        # track how many steps gradient descent takes
        count1 = 0

        if indicator == 'latent_covariances':
            grad_func = self.variational_gradient
            update_set = self.latent_Sigmas
            resize = lambda x: np.diag(np.diag(x.reshape(self.q, self.q))) # this zeros all off-diagonal elements
            parameter = 'Si'
        else:
            grad_func = self.partition_gradient
            
        if indicator in ['nu', 'beta']:
            parameter = indicator

            if indicator == 'nu':
                par_old = self.nu
            else:
                par_old = self.beta

            par_new  = par_old + self.step_size*grad_func(parameter=parameter)

            # update until squared error is less than tolerance
            while LA.norm(par_old - par_new, 2)**2 > 0.01:
                par_old = par_new

                par_new = par_old + self.step_size*grad_func(parameter=parameter)

                count1 += 1
                if count1 > self.max_iterations:
                    # print('Max iterations reached before tolerance...\n')
                    break

            return 0

        # optimize latent parameters using gradient descent
        # update each parameter in the 'update_set'
        for index in range(self.N):
            par_old = update_set[index].flatten()
            par_new  = par_old + self.step_size*grad_func(index=index, parameter=parameter)
        
            # update until squared error is less than tolerance
            while LA.norm(par_old - par_new, 2)**2 > 0.01:
                par_old = par_new

                par_new = par_old + self.step_size*grad_func(index=index, parameter=parameter)

                count1 += 1
                if count1 > self.max_iterations:
                    # print('Max iterations reached before tolerance...\n')
                    break
                
            # update the latent parameter with new optimum
            update_set[index] = resize(par_new)

        return 0

    def optimize_variational_means(self, j):
        """ Optimize variational means with constraint
            nu^Tmi + beta >= 0 for j = 1 
            nu^Tmi + beta < 0 for j = 2   i.e. -nu^Tmi - beta > 0
        
        """
        assert j in [0, 1]

        index_set = self.index_sets[j].copy()

        nu = self.nu.flatten()
        beta = self.beta

        constraint = {'type': 'ineq', 'fun': lambda x: (-1)**j*(np.dot(nu, x) + beta)/np.sqrt(np.dot(nu, nu))}
        bnd = ((-100, 100), (-100, 100))

        for i in index_set:
            mi = self.latent_means[i].flatten() # separate mi from the rest of the means

            # need to pass in all latent_means and latent_Sigmas to compute the full log-likelihood
            _opt_mi = minimize(
                obj,
                mi,
                method='SLSQP', # SLSQP
                constraints=constraint,
                bounds=bnd,
                args=(self, i, j, 'mi')
            )

            self.latent_means[i] = _opt_mi.x.reshape(-1, 1) # place the updated mean back with the others

    def optimize_nu(self):
        """ Optimize partition parameter vector nu with conjugate gradient descent. """

        _opt_nu = minimize(
            obj,
            self.nu,
            method='CG',
            args=(self, -1, -1, 'nu')
        )

        nu = _opt_nu.x.reshape(-1, 1)

        # normalize
        # self.nu /= LA.norm(nu, 2)
        self.nu = nu

    def optimize_beta(self):
        """ Optimize partition parameter beta with conjugate gradient descent. """
        _opt_beta = minimize(
            obj,
            self.beta,
            method='CG',
            args=(self, -1, -1, 'beta')
        )

        self.beta = _opt_beta.x.item()

    def update_mu(self, j):
        """ Get terms for mu_1 or mu_2 """
        assert j in [0, 1]
        
        B = self.model_maps[j]
        err_func = self.error_funcs[j]
        index_set = self.index_sets[j]
        
        _di = [self.deltas[i] for i in index_set]
        _mi_terms = [np.matmul(B, self.latent_means[i]) for i in index_set]
        _yi = [self.Y[i] for i in index_set]

        _reciprocal = sum([err_func(di) for di in _di])
        _sum_term = sum([err_func(di)*(yi - mi_term) for di, yi, mi_term in zip(_di, _yi, _mi_terms)])

        return _sum_term/_reciprocal

    def update_B(self, j: int):
        """ Get the terms involving either B1 or B2, 
            depending on whether we are in Omega plus
            or Omega minus
        """
        assert j in [0, 1]

        mu = self.model_means[j]
        error_func = self.error_funcs[j]
        index_set = self.index_sets[j]
        deltas = [self.deltas[i] for i in index_set]
        means = [self.latent_means[i] for i in index_set]
        Sigmas = [self.latent_Sigmas[i] for i in index_set]
        Y = [self.Y[i] for i in index_set]
        
        _not_inverted = sum([np.matmul(yi-mu, mi.reshape(1, -1)) for yi, mi in zip(Y, means)])

        # compute terms for the part that gets inverted
        _si = [Si[-1, -1] for Si in Sigmas]
        
        # the (qxq)th basis vector for qxq matrices
        _Mq = np.zeros((self.q, self.q))
        _Mq[-1, -1] = 1
        
        _to_invert_term1 = [(-1)**(j+1)*di*np.exp(-di**2/2)*si*_Mq for di, si in zip(deltas, _si)]
        _to_invert_term2 = [
            SQRT_PI_OVER_2*error_func(di)*(
                Si + np.matmul(
                    mi,
                    mi.reshape(1, -1)
                )
            ) for di, Si, mi in zip(deltas, Sigmas, means)
        ]

        # compute the inverse
        _to_invert = sum([item1 + item2 for item1, item2 in zip(_to_invert_term1, _to_invert_term2)])

        _inverted_part = LA.inv(_to_invert)

        return np.matmul(_not_inverted, _inverted_part)
    
    def M_step(self, j: int):
        """ Update the model parameters based on the maximum likelihood estimates 
            for the parameters either over I1 or I2.

            This comprises the M-step of the EM algorithm.
        """
        # update the model parameters
        mu = self.update_mu(j=j)
        B = self.update_B(j=j)

        # update constant (2pi)^(-1/2)/(2sigma^2)
        self._const = -1/(ROOT2PI*2*self.sigma2)
        self._scalars = 0.5*self.N*(self.q - self.p*np.log(TWOPI*self.sigma2))

        # optimize variational means
        self.optimize_variational_means(j=j)

        return mu, B

    def loglik(self, j: int):
        """ Compute the VLB of the loglikelihood for either I1 or I2. """
        _index_set = self.index_sets[j]

        _inner_terms = get_omega_terms(j=j, index_set=self.index_sets[j], deltas=self.deltas, problem=self)
        _global_terms = 0.5*self.N*(self.q - self.p*np.log(TWOPI*self.sigma2))
        _determinants_minus_traces = [np.prod(np.diag(self.latent_Sigmas[i])) - np.trace(self.latent_Sigmas[i]) for i in _index_set]
        _miTmi_terms = sum([np.sqrt(sum(self.latent_means[i]**2)).item() for i in _index_set])
        _global_terms += 0.5*(sum(_determinants_minus_traces) - _miTmi_terms)

        _result = _global_terms + self._const*_inner_terms

        return _result
    
    def em_algorithm(self, j: int):
        """ Perform the EM algorithm over either I1 or I2. Essentially this
            treats the VLB as two separate problems.
        """

        current_loglik = self.loglik(j=j)
        previous_loglik = 2*current_loglik
        em_iterations = 0

        # keep performing EM until the change in loglikelihood is less than the tolerance
        print('Optimizing model parameters over I{} using the EM algorithm...'.format(j))

        while np.abs(current_loglik - previous_loglik) > self.em_tolerance:
            # update the model and variational parameters
            mu, B = self.M_step(j=j)

            self.model_means[j] = mu
            self.model_maps[j] = B
            
            previous_loglik = current_loglik

            # E-step
            current_loglik = self.loglik(j)
            self._loglik_results[j].append(current_loglik)

            em_iterations += 1

            if em_iterations > self.max_iterations:
                print('Maximum iterations reached...')
                break

        self._loglik_results[j].append(self.loglik(j=j))
        # update the index sets
        # self.update_index()

        print('Optimal parameters for set I{} reached in {} iterations.'.format(j, em_iterations+1))
        print('Log-likelihood at the optimum: {}'.format(self._loglik_results[j][-1]))

    def optimize_model(self):
        """ Perform the EM algorithm over the model parameters B1, B2, and sigma2 
            and latent means and covariance matrices
        """

        self.em_algorithm(j=0)
        self.em_algorithm(j=1)

    def visualize(self):
        # make two nxq matrices of estimated latent means
        M1 = np.vstack([self.latent_means[i].flatten() for i in self.I1])
        M2 = np.vstack([self.latent_means[i].flatten() for i in self.I2])

        # get the raw latent positions
        P1 = np.matmul(M1, self.B1.T)
        P2 = np.matmul(M2, self.B2.T)

        # orthogonalize the latent positions
        # pca1 = PCA(n_components=3)
        # pca2 = PCA(n_components=3)
        # U1 = pca1.fit_transform(P1)
        # U2 = pca2.fit_transform(P2)

        ax = plt.axes()
        ax2 = plt.axes(projection='3d')

        ax.scatter(M1[:, 0], M1[:, 1], color='red')
        ax2.scatter3D(P1[:, 0], P1[:, 1], P1[:, 2], color='blue')
        plt.show()

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
    # fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(6, 12))

    index1 = [i for i in range(problem.N) if problem.Y[i][0].item() >= 0]
    index2 = [i for i in range(problem.N) if problem.Y[i][0].item() < 0]

    obs1 = np.array([problem.Y[i] for i in index1])
    obs2 = np.array([problem.Y[i] for i in index2])

    x1 = obs1[:, 0]
    y1 = obs1[:, 1]
    z1 = obs1[:, 2]

    x2 = obs2[:, 0]
    y2 = obs2[:, 1]
    z2 = obs2[:, 2]

    # plot the observations
    # fig2 = plt.figure(figsize=(6, 6))
    # ax3 = fig2.add_subplot(projection='3d')
    # ax3.scatter3D(x1, y1, z1, color='#d82f49ff')
    # ax3.scatter3D(x2, y2, z2, color='#6a2357ff')
    # ax3.scatter3D(x1[2], y1[2], z1[2], color='blue', marker='X', s=100)
    # ax3.azim = -80
    # ax3.elev = 16
    # ax3.set_title('Observations')
    # fig2.savefig('results/example2_observations.png', bbox_inches='tight')

    y1 = [problem.Y[i] for i in problem.I1]
    y2 = [problem.Y[i] for i in problem.I2]

    def sample_hidden_given_visible(
        weight_ml : np.array, 
        mu_ml : np.array,
        visible_samples : np.array
        ) -> np.array:

        var_ml = 1.

        q = weight_ml.shape[1]
        m = np.transpose(weight_ml) @ weight_ml + var_ml * np.eye(q)

        cov = var_ml * np.linalg.inv(m)
        act_hidden = []
        for data_visible in visible_samples:
            mean = np.linalg.inv(m) @ np.transpose(weight_ml) @ (data_visible - mu_ml)
            sample = np.random.multivariate_normal(mean.flatten(), cov, size=1) # this needs to be my posterior distribution
            act_hidden.append(sample[0])
        
        return np.array(act_hidden)

    hidden1 = np.vstack([sample_hidden_given_visible(problem.B1, problem.mu1, yi) for yi in y1])
    hidden2 = np.vstack([sample_hidden_given_visible(problem.B2, problem.mu2, yi) for yi in y2])

    plt.scatter(hidden1[:, 0], hidden1[:, 1], color='red')
    plt.scatter(hidden2[:, 0], hidden2[:, 1], color='blue')
    plt.show()
    # ax2 = plt.axes(projection='3d')
    # xx1 = [item[0] for item in problem.positions1]
    # yy1 = [item[1] for item in problem.positions1]
    # zz1 = [item[2] for item in problem.positions1]
    # xx2 = [item[0] for item in problem.positions2]
    # yy2 = [item[1] for item in problem.positions2]
    # zz2 = [item[2] for item in problem.positions2]
    # ax2.scatter3D(xx1, yy1, zz1, color='blue')
    # ax2.scatter3D(xx2, yy2, zz2, color='red')
    # plt.show()

    # # plot the latent space - piecewise PCA
    # # reshape the latent means for plotting
    # m = np.vstack([row.reshape(1, -1) for row in problem.latent_means])
    # lat1 = m[index1]
    # lat2 = m[index2]
    # ax1.scatter(lat1[:, 0], lat1[:, 1], color='#d82f49ff')
    # ax1.scatter(lat2[:, 0], lat2[:, 1], color='#6a2357ff')
    # ax1.scatter(lat1[2, 0], lat1[2, 1], color='blue', marker='X', s=100)
    # ax1.set_aspect('equal')
    # ax1.set_title('Estimated positions in latent space - piecewise PCA')

    # # plot the optimal partition
    # x = np.linspace(np.min(lat1[:, 0]), np.max(lat1[:, 0]), 25)
    # y = [0]*25 #-1/(problem.nu[1].item())*(problem.nu[0].item()*x + problem.beta)
    # ax1.plot(x, y, color='black')

    # # plot the latent space - classical PCA
    # if pca_coords is not None:
    #     x1 = [item[0] for item in pca_coords[problem.I1]]
    #     y1 = [item[1] for item in pca_coords[problem.I1]]
    #     x2 = [item[0] for item in pca_coords[problem.I2]]
    #     y2 = [item[1] for item in pca_coords[problem.I2]]

    #     ax2.scatter(x1, y1, color='#d82f49ff')
    #     ax2.scatter(x2, y2, color='#6a2357ff')
    #     ax2.set_aspect('equal')
    #     ax2.set_title('Estimated positions in latent space - PCA')

    # plt.show()
    # fig1.savefig("results/example2_result.png", bbox_inches='tight')

def example(n_obs, data_type, latent_dimension=2):
    """ Compute Example 1 or 2 """

    p = Problem(n_obs=n_obs, data_type=data_type, latent_dimension=latent_dimension, em_tolerance=1e-3, max_iterations=100)
    print(p)

    p.optimize_model()
    # print('\nPosterior model mean parameters...')
    # print(p.mu1, p.mu2)
    # print('\nPosterior transformations...')
    # print(p.B1, p.B2)

    # for comparison, let's do classical PCA with the data
    pca = PCA(n_components=2)
    coords = p.data.coords
    pca_coords = pca.fit_transform(coords)

    sns.set_theme()
    p.visualize()

    # now plot piecewise pca and classical PCA
    # if data_type == 'example2':
    #     plot_example2(p, pca_coords=pca_coords)

if __name__ == '__main__':
    # Example 2
    example(100, 'example2')
