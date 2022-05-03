import numpy as np
import numpy.linalg as LA
from scipy.special import erf, erfc
from demodata import DemoData
from config import *
import nlopt
from alive_progress import alive_bar
import matplotlib.pyplot as plt

class Problem:
    def __init__(self, n_obs, data_type='helix', latent_dimension=2, timeout=15, tolerance=1e-2, max_iterations=25):
        self.q = latent_dimension
        self.N = n_obs  # number of observations
        self.timeout = timeout

        # define step size for gradient ascent
        self.step_size = 0.01

        # setup the observations
        if data_type == 'helix':
            self.obs = DemoData(kind='helix', n_obs=self.N, mean=None, covariance=1.1)
            
        elif data_type == 'hinge':
            cov = np.diag(np.random.randn(2)**2)
            self.obs = DemoData(kind='hinge', n_obs=self.N, mean=[0, 0], covariance=cov)

        elif data_type == 'mixture':
            cov = np.diag(np.random.randn(2)**2)
            self.obs = DemoData(kind='mixture', n_obs=self.N, mean=[0, 0], covariance=cov)

        # apply rotation in 3D
        # TODO: make alpha, beta, gamma parameters for the rotation
        self.obs.rotate()

        # get the coordinates from the object
        self.coords = self.obs.coords

        self.p = self.coords.T.shape[0]   # dimension of the observation space
        
        # stopping criteria for optimization algorithms
        self.tolerance = tolerance   # tolerance for the loglikelihood in the EM algorithm
        self.max_iterations = max_iterations

        # initialize a set of means and standard deviations for the latent variables
        self.latent_means = [np.random.randn(self.q, 1) for _ in range(self.N)]
        self.latent_variances = [np.ones(self.q) for _ in range(self.N)]
        self.latent_Sigmas = [np.diag(var) for var in self.latent_variances]

        # set starting values for sigma2, mu1, mu2, B_1, B_2
        self.mu1 = np.mean(self.coords.T, axis=1).reshape(1, -1) 
        self.mu2 = self.mu1.copy()
        self.sigma2 = np.random.rand()       # set to random positive number
        self.B1 = np.random.randn(self.p, self.q)
        self.B2 = np.random.randn(self.p, self.q)

        # indexes that correspond to the two separate subsets of the latent data
        self.I1 = [index for index, mi in enumerate(self.latent_means) if mi[-1] >= 0]
        self.I2 = [index for index, mi in enumerate(self.latent_means) if mi[-1] < 0]

        # I want the observations to be 1xp arrays for later computations
        self.Y = [yi.reshape(1, -1) for yi in self.coords]

        # setup the optimizers for the latent means and covariance matrices
        self.opt_means = nlopt.opt(nlopt.LD_MMA, self.q)  # MMA is method of moving asymptotes

        # this sets the lower and upper bounds for the unconstrained optimization of the latent means
        self.opt_means.set_lower_bounds([-float('inf')]*self.q)
        self.opt_means.set_upper_bounds([float('inf')]*self.q)

        # set the error tolerance of the optimizer
        self.opt_means.set_xtol_rel(self.tolerance)

        self.opt_means.set_max_objective(self.LL_mean)

        # optimize wrt the latent covariance matrices
        self.opt_Sigmas = nlopt.opt(nlopt.LD_MMA, self.q*self.q)
        # opt_Sigmas.set_xtol_rel(tolerance)
        self.opt_Sigmas.set_max_objective(self.LL_Sigma)

        # constrain covariance matrices to be diagonal
        # lower_bounds_diag = -float('inf')*np.ones(self.q)
        upper_bounds_diag = float('inf')*np.ones(self.q)
        # lower_bounds = np.diag(lower_bounds_diag).ravel()
        lower_bounds = np.zeros(self.q**2)
        upper_bounds = np.diag(upper_bounds_diag).ravel()

        self.opt_Sigmas.set_lower_bounds(lower_bounds)
        self.opt_Sigmas.set_upper_bounds(upper_bounds)

        # set a timeout for the optimization
        self.opt_Sigmas.set_maxtime(maxtime=self.timeout)

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

    def plot(self):
        """ Plot the dataset """
        plt.figure()

        ax1 = plt.axes(projection='3d')       

        # extract the individual axis coordinates
        self.x = self.coords[:, 0]
        self.y = self.coords[:, 1]
        self.z = self.coords[:, 2]

        # first plot the latent points in Omega_plus
        ax1.scatter3D(self.x, self.y, self.z, color='red')

        plt.show()

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

    def LL_Sigma(self, Sigma, gradient_Sigma):
        """ Optimize the log-likelihood of the variational lower bound wrt  
            a single latent covariance matrix given by Sigma.
        """

        if gradient_Sigma.size > 0:
            gradient_Sigma[:] = self.gradient_wrt_latent_Sigma(Si=Sigma)

        self.latent_Sigmas[self.index] = Sigma.reshape(self.q, self.q)

        loglik = self.loglikelihood()
        return loglik

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
            
    def optimize_Sigma(self):
        """ Perform the optimization with respect to the latent covariance matrices. """

        self.opt_Sigmas.set_maxeval(self.max_iterations)

        print('\nOptimizing the log-likelihood with respect to latent covariance matrices...')
        # with alive_bar(N) as bar:
        for index, Si in enumerate(self.latent_Sigmas):
            # NLOPT only wants flat inputs
            Si = Si.ravel()

            self.index = index

            # TODO: make this work! It's not updating the Sigmas and I'm getting singular matrices
            
            xi = self.opt_Sigmas.optimize(Si)

            # The problem seems to be taking place where the Sigmas are updated.

            # update the latent covariance matrices with the new optimum
            self.latent_Sigmas[index] = xi.reshape(self.q, self.q)

    def LL_mean(self, mean, gradient_mean):
        """ Compute the log-likehood of the variational lower bound for the problem. 

            :mean: np.array
                contains the current value of the latent mean with respect to which
                we are optimizing

            :gradient_mean: np.array
                contains the gradient of the loglikelihood
                with respect to a single latent mean
        """

        if gradient_mean.size > 0:
            # update the gradient
            gradient_mean[:] = self.gradient_wrt_latent_mean(mi=mean)

        self.latent_means[self.index] = mean.reshape(-1, 1)

        loglik = self.loglikelihood()
        return loglik

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

    def optimize_means(self):
        """ Perform the optimization with respect to the latent means. """

        self.opt_means.set_maxeval(self.max_iterations)

        print('\nOptimizing log-likelihood with respect to latent means...')
        with alive_bar(self.N) as bar:
            for index, mi in enumerate(self.latent_means):            
                mi = mi.flatten()

                self.index = index

                xi = self.opt_means.optimize(mi)
                
                # update the latent means with the new optimum
                self.latent_means[index] = xi.reshape(-1, 1)
                bar()

    def gradient_descent(self, indicator: str):
        """ Perform gradient descent/ascent """
        pass

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

        # optimize latent parameters using gradient ascent
        # TODO: get progress bar working for gradient ascent
        for index, mi in enumerate(self.latent_means):
            self.index = index

            loglik_old = self.LL_mean(mi, gradient_mean=np.empty((0, 0)))

            mi_old = mi.flatten()
            mi_new  = mi_old - self.step_size*self.gradient_wrt_latent_mean(mi_old)

            loglik_new = self.LL_mean(mi_new, gradient_mean=np.empty(0))
            
            count1 = 0
            # update until absolute error is less than tolerance
            while np.abs((loglik_old-loglik_new)/loglik_old) > self.tolerance:
                mi_old = mi_new
                loglik_old = loglik_new.copy()

                mi_new = mi_old - self.step_size*self.gradient_wrt_latent_mean(mi_old)
                loglik_new = self.LL_mean(mi_new, gradient_mean=np.empty(0))

                count1 += 1
                if count1 > self.max_iterations:
                    break

                # decrease the step size when we're getting close to a solution
                # if np.abs(loglik_old-loglik_new) < 2*self.tolerance:
                #     self.step_size *= .5
                
            # update the latent means with new optimum
            self.latent_means[index] = mi_new.reshape(-1, 1)
        
        for index, Si in enumerate(self.latent_Sigmas):
            # now work on the latent Sigmas
            # find out if the index is in either of the Omega plus or Omega minus index sets
            self.index = index
            
            loglik_old = self.LL_Sigma(Si, gradient_Sigma=np.empty(0))          

            Si_old = Si.flatten()
            
            Si_new = Si_old + self.step_size*self.gradient_wrt_latent_Sigma(Si=Si_old)
            loglik_new = self.LL_Sigma(Si_new, gradient_Sigma=np.empty(0))
            
            while np.abs(loglik_old-loglik_new) > self.tolerance:
                Si_old = Si_new
                loglik_old = loglik_new

                Si_new = Si_old + self.step_size*self.gradient_wrt_latent_Sigma(Si=Si_old)
                
                loglik_new = self.LL_Sigma(Si_new, gradient_Sigma=np.empty(0))

            # update latent Sigmas with new optimum
            self.latent_Sigmas[index] = np.diag(np.diag(Si_new.reshape(self.q, self.q))) # this zeros all off-diagonal elements

    def optimize_model(self):
        """ Perform the EM algorithm over the model parameters B1, B2, and sigma2 
            and latent means and covariance matrices
        """

        # keep track of loglikelihood
        L_new = self.loglikelihood()
        L_old = 0

        self._loglik_results = [L_new]

        # keep performing EM until the change in loglikelihood is less than the tolerance
        print('Optimizing model parameters using the EM algorithm...')
        for j in range(20):
            L_old = L_new
            
            # update the model parameters
            self.M_step()
            
            L_new = self.loglikelihood()
            self._loglik_results.append(L_new)

        print('\nOptimizing variational parameters...')
        # optimize latent parameters using Method of Moving Asymptotes
        self.optimize_means()
        self.optimize_Sigma()

        self.loglik = self.loglikelihood()
        self._loglik_results.append(self.loglik.copy())

        print('Optimal parameters reached.')
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

    def plot_result(self):
        """ Plot the latent means transformed through B1 and B2 """

        ax1 = plt.axes(projection='3d')

        coords1 = [self.coords[index] for index in self.I1]       
        coords2 = [self.coords[index] for index in self.I2]

        # extract the individual axis coordinates
        self.x1 = [point[0] for point in coords1]
        self.y1 = [point[1] for point in coords1]
        self.z1 = [point[2] for point in coords1]

        self.x2 = [point[0] for point in coords2]
        self.y2 = [point[1] for point in coords2]
        self.z2 = [point[2] for point in coords2]

        # first plot the observation points
        ax1.scatter3D(self.x1, self.y1, self.z1, color='red')
        ax1.scatter3D(self.x2, self.y2, self.z2, color='orange')

        # plot the latent points in Omega plus/minus
        ax1.scatter3D(self.result_x1, self.result_y1, self.result_z1, color='blue')
        ax1.scatter3D(self.result_x2, self.result_y2, self.result_z2, color='black')

        # ax2 = plt.axes(projection='3d')

        plt.show()

if __name__ == '__main__':
    
    p = Problem(n_obs=201, data_type='hinge', latent_dimension=2, tolerance=1e-3)  
    
    print(p)
    # p.plot()
    
    p.optimize_model()
    
    p.get_result()
    p.plot_result()
    