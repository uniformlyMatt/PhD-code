import numpy as np
import numpy.linalg as LA
from scipy.special import erf, erfc
from demodata import DemoData
from config import *
import nlopt
from alive_progress import alive_bar
import matplotlib.pyplot as plt

class Problem:
    def __init__(self, n_obs, data_type='helix', latent_dimension=2, timeout=15, tolerance=1e-2):
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
        self.obs = self.coords.T

        self.p = self.obs.shape[0]   # dimension of the observation space
        
        self.tolerance = tolerance   # tolerance for the loglikelihood in the EM algorithm

        # initialize a set of means and standard deviations for the latent variables
        self.latent_means = [np.random.randn(self.q, 1) for _ in range(self.N)]
        self.latent_variances = [np.abs(np.random.randn(self.q))+1 for _ in range(self.N)]
        self.latent_Sigmas = [np.diag(var) for var in self.latent_variances]

        # these are the 's' parameters when nu=e_q, beta=0
        # ss = [-mi[-1]*np.sqrt(si[-1]) for mi, si in zip(latent_means, latent_variances)]

        # set starting values for sigma2, mu, B_1, B_2, g_1, and g_2
        self.mu = np.mean(self.obs, axis=1).reshape(1, -1) # the optimal value for mu is the empirical mean of the data
        self.sigma2 = np.random.rand()       # set to random positive number
        self.B1 = np.random.randn(self.p, self.q)
        self.B2 = np.random.randn(self.p, self.q)

        # indexes that correspond to the two separate subsets of the latent data
        self.I1 = [index for index, mi in enumerate(self.latent_means) if mi[-1] >= 0]
        self.I2 = [index for index, mi in enumerate(self.latent_means) if mi[-1] < 0]

        # these are nuisances; set them to 1 for now
        self.g1 = 1
        self.g2 = 1

        # I want the observations to be 1xp arrays for later computations
        self.Y = [yi.reshape(1, -1) for yi in self.obs.T]

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
        result += 'Initial Log-likelihood: {}'.format(self.loglik)
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

    def loglikelihood(self):
        """ Compute the log-likelihood function.
            Compute the summation terms for the log-likelihood. 
        """
        # update the latent variances based on the updated latent covariances matrices
        self.latent_variances = [np.diag(Si) for Si in self.latent_Sigmas]

        # these are the 's' parameters when nu=e_q, beta=0
        ss = [-mi[-1]*np.sqrt(si[-1]) for mi, si in zip(self.latent_means, self.latent_variances)]

        # add up the terms that are valid over the entire index 1 <= i <= N
        determinants = [np.prod(np.diag(Sigma)) for Sigma in self.latent_Sigmas]
        traces = [np.trace(Sigma) for Sigma in self.latent_Sigmas]
        miTmi = [sum(mi**2)**2 for mi in self.latent_means]
        yi_minus_mu_quadratic = [np.sum(yi-self.mu)**2/self.sigma2 for yi in self.coords]

        global_terms = [0.5*(-tr - mi_norm + np.log(det) - y_quad) for tr, det, mi_norm, y_quad in zip(traces, determinants, miTmi, yi_minus_mu_quadratic)]

        # now for the terms in Omega_plus
        # update the index sets
        self.I1 = [index for index, mi in enumerate(self.latent_means) if mi[-1] >= 0]
        self.I2 = [index for index, mi in enumerate(self.latent_means) if mi[-1] < 0]

        def get_omega_terms(plus=True):
            """ Compute the summation terms over Omega plus or Omega minus """
            if plus:
                B =  self.B1    
                g = self.g1
                error_func = lambda x: erfc(x)
                exp_func = lambda x: np.exp(x)
                index_set = self.I1

            else:
                B = self.B2
                g = self.g2
                error_func = lambda x: erf(x) + 1
                exp_func = lambda x: -np.exp(x)
                index_set = self.I2

            BTB = np.matmul(B.T, B)

            _yi_terms = [
                -2*np.matmul(
                    np.matmul(
                        self.coords[index]-self.mu,
                        B
                    ),
                    self.latent_means[index]
                ) for index in index_set
            ]
            _g_terms = [
                g*(SQRT_PI_OVER_2*error_func(ss[index]/ROOT2) + ss[index]*exp_func(-ss[index]**2/2)) for index in index_set
            ]
            _trace_terms = [
                np.trace(
                    np.matmul(
                        BTB,
                        self.latent_Sigmas[index]
                    )
                ) for index in index_set
            ]
            _quadratic_terms = [
                np.matmul(
                    self.latent_means[index].T,
                    np.matmul(
                        BTB,
                        self.latent_means[index]
                    )
                )*SQRT_PI_OVER_2*error_func(ss[index]/ROOT2) for index in index_set
            ]
            _terms = [
                (-1/(2*self.sigma2))*(item1 + TWOPI**(0.5-self.q)*(item2*item3 + item4)) for item1, item2, item3, item4 in zip(
                    _yi_terms,
                    _g_terms,
                    _trace_terms,
                    _quadratic_terms
                )
            ]

            return _terms

        self.omega_plus_terms = get_omega_terms(plus=True)
        self.omega_minus_terms = get_omega_terms(plus=False)

        # finally, compute the scalars that are independent of the data and latent variables
        scalars = self.N*self.q/2 - self.N*self.p*np.log(TWOPI*self.sigma2)/2

        # add all of the terms together
        total = np.sum(global_terms) + (np.sum(self.omega_plus_terms) + np.sum(self.omega_minus_terms))

        return total + scalars

    def LL_Sigma(self, Sigma, gradient_Sigma):
        """ Optimize the log-likelihood of the variational lower bound wrt  
            the latent covariance matrix Sigma.
        """

        # TODO: calculate gradient over the correct index set
        if gradient_Sigma.size > 0:
            gradient_Sigma[:] = self.gradient_wrt_latent_Sigma(Si=Sigma)

        self.latent_Sigmas[self.index] = Sigma.reshape(self.q, self.q)

        loglik = self.loglikelihood()
        return loglik

    def gradient_wrt_latent_Sigma_omega_plus(self, Si):
        """ Computes the gradient of the loglikelihood with 
            respect to a single latent covariance matrix
            over the index set corresponding to Omega plus.
        """
        # extract the square root of the q,q element from the covariance matrix while it's still flattened
        s_iq = np.sqrt(Si[-1])

        # reshape the latent covariance matrix from a flat vector to a matrix
        Si = Si.reshape(self.q, self.q)

        # since Si is a diagonal matrix, we know the inverse is just the reciprocal
        Si_inverse = np.diag(1/np.diag(Si))

        mi = self.latent_means[self.index]
        delta_i = -mi[-1]*s_iq

        scalars = -0.5*(np.eye(self.q) + Si_inverse)

        # derivative of scalar delta_i wrt Sigma_i
        ddelta_idSi = np.zeros((self.q, self.q))
        ddelta_idSi[-1][-1] = -0.5*mi[-1]/s_iq

        # precompute some of the required terms involving B1, B2
        B1TB1 = np.matmul(self.B1.T, self.B1)

        inner_exp_terms = np.exp(-delta_i**2/2)*(
            delta_i*B1TB1 - \
            ddelta_idSi*(delta_i**2*(np.trace(np.matmul(B1TB1, Si)) + np.matmul(np.matmul(mi.T, B1TB1), mi)))
        )

        inner_erf_terms = SQRT_PI_OVER_2*erfc(delta_i/ROOT2)*B1TB1

        result = scalars - 1/(2*self.sigma2)*TWOPI**(1/2-self.q)*(inner_exp_terms + inner_erf_terms)

        return result.ravel()

    def gradient_wrt_latent_Sigma_omega_minus(self, Si):
        """ Computes the gradient of the loglikelihood with 
            respect to a single latent covariance matrix
            over the index set corresponding to Omega minus.
        """
        # extract the square root of the q,q element from the covariance matrix while it's still flattened
        s_iq = np.sqrt(Si[-1])

        # reshape the latent covariance matrix from a flat vector to a matrix
        Si = Si.reshape(self.q, self.q)

        # since Si is a diagonal matrix, we know the inverse is just the reciprocal
        Si_inverse = np.diag(1/np.diag(Si))

        mi = self.latent_means[self.index]
        delta_i = -mi[-1]*s_iq

        scalars = -0.5*(np.eye(self.q) + Si_inverse)

        # derivative of scalar delta_i wrt Sigma_i
        ddelta_idSi = np.zeros((self.q, self.q))
        ddelta_idSi[-1][-1] = -0.5*mi[-1]/s_iq

        # precompute some of the required terms involving B1, B2
        B2TB2 = np.matmul(self.B2.T, self.B2)

        inner_exp_terms = np.exp(-delta_i**2/2)*(
            delta_i*B2TB2 - \
            ddelta_idSi*(delta_i**2*(np.trace(np.matmul(B2TB2, Si)) + np.matmul(np.matmul(mi.T, B2TB2), mi)))
        )

        inner_erf_terms = SQRT_PI_OVER_2*(erfc(delta_i/ROOT2)+1)*B2TB2

        result = scalars - 1/(2*self.sigma2)*TWOPI**(1/2-self.q)*(inner_exp_terms + inner_erf_terms)

        return result.ravel()

    def optimize_Sigma(self, max_iterations=1):
        """ Perform the optimization with respect to the latent covariance matrices. """

        self.opt_Sigmas.set_maxeval(max_iterations)

        print('\nOptimizing the log-likelihood with respect to latent covariance matrices...')
        # with alive_bar(N) as bar:
        for index, Si in enumerate(self.latent_Sigmas):
            # NLOPT only wants flat inputs
            Si = Si.ravel()

            self.index = index

            # TODO: make this work! It's not updating the Sigmas and I'm getting singular matrices

            print(Si)
            
            xi = self.opt_Sigmas.optimize(Si)

            # The problem seems to be taking place where the Sigmas are updated.

            # update the latent covariance matrices with the new optimum
            self.latent_Sigmas[index] = xi.reshape(self.q, self.q)

            print('Maximum value: {}'.format(self.opt_Sigmas.last_optimum_value()))
            print('Optimum at {}\n'.format(xi))

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
            respect to a single latent mean.
        """
        # these are the 's' parameters when nu=e_q, beta=0
        Sigma = self.latent_Sigmas[self.index]

        si = -mi[-1]*np.sqrt(Sigma.flatten()[-1])
        
        yi = self.Y[self.index]

        B1_plus_B2 = self.B1 + self.B2

        b0 = -mi.reshape(-1, 1) + 1/(2*self.sigma2)*np.matmul(B1_plus_B2.T, (yi-self.mu).T)

        b1 = (SQRT_PI_OVER_2-1-si**2)*np.exp(-si**2/2)*self.g1*np.trace(
            np.matmul(
                np.matmul(
                    self.B1.T, self.B1
                ),
                Sigma
            )
        )

        b2 = si*np.exp(-si**2/2)*self.g2*np.trace(
            np.matmul(
                np.matmul(
                    self.B2.T,
                    self.B2
                ),
                Sigma
            )
        )

        B1TB1 = np.matmul(self.B1.T, self.B1)
        B2TB2 = np.matmul(self.B2.T, self.B2)

        b3 = np.exp(-si**2/2)*np.matmul(
            mi.T,
            np.matmul(
                B2TB2 - B1TB1,
                mi
            )
        )

        b4 = ROOT2PI*np.matmul(
            mi,
            erfc(si/ROOT2)*B1TB1 + (erf(si/ROOT2)+1)*B2TB2
        ).reshape(-1, 1)

        result = b0 + b4

        # update the qth element of the derivative with the corresponding derivative elements
        result[-1] += (TWOPI)**(1/2-self.q)/(2*self.sigma2)*(b1 + b2 + b3)

        return result.flatten()

    def optimize_means(self, max_iterations=1):
        """ Perform the optimization with respect to the latent means. """

        self.opt_means.set_maxeval(max_iterations)

        print('\nOptimizing the log-likelihood with respect to latent means...')
        # with alive_bar(N) as bar:
        # optimize wrt the latent means
        print('\nOptimizing log-likelihood with respect to latent means...')
        with alive_bar(self.N) as bar:
            for index, mi in enumerate(self.latent_means):            
                mi = mi.flatten()

                self.index = index

                print(mi)

                xi = self.opt_means.optimize(mi)
                
                # update the latent means with the new optimum
                self.latent_means[index] = xi.reshape(-1, 1)
                # print('Maximum value: {}'.format(self.opt_means.last_optimum_value()))
                # print('Optimum at {}\n'.format(xi)) 
                bar()  
    
    def M_step(self):
        """ Update the model parameters based on the maximum likelihood estimates 
            for the parameters.

            This comprises the M-step of the EM algorithm.
        """
        ss = [-mi[-1]*np.sqrt(si[-1]) for mi, si in zip(self.latent_means, self.latent_variances)]
        
        # update the linear transformations B1 and B2
        def get_B_terms(plus=True):
            """ Get the terms involving either B1 or B2, 
                depending on whether we are in Omega plus
                or Omega minus
            """
            if plus:
                g = self.g1
                error_func = lambda x: erfc(x)
                exp_func = lambda x: np.exp(x)
                index_set = self.I1
            else:
                g = self.g2
                error_func = lambda x: erf(x) + 1
                exp_func = lambda x: -np.exp(x)
                index_set = self.I2

            to_invert_part1 = [
                g*self.latent_Sigmas[index]*(SQRT_PI_OVER_2*error_func(ss[index]/ROOT2) + ss[index]*exp_func(-ss[index]**2/2)) for index in index_set
            ]
            to_invert_part2 = [SQRT_PI_OVER_2*np.matmul(self.latent_means[index], self.latent_means[index].T)*error_func(ss[index]/ROOT2) for index in index_set]
            inverted_part = sum([p1 + p2 for p1, p2 in zip(to_invert_part1, to_invert_part2)])

            return LA.inv(inverted_part)

        inverted_part_omega_plus = get_B_terms(plus=True)
        inverted_part_omega_minus = get_B_terms(plus=False)

        _y_minus_mu_mi_terms = [
            np.matmul(
                (yi - self.mu).reshape(-1, 1),
                mi.T
            ) for yi, mi in zip(self.coords, self.latent_means)
        ]
        y_minus_mu_mi_omega_plus = sum(
            [
                _y_minus_mu_mi_terms[index] for index in self.I1
            ]
        )
        y_minus_mu_mi_omega_minus = sum(
            [
                _y_minus_mu_mi_terms[index] for index in self.I2
            ]
        )

        _scalar_constant = (2*TWOPI)**(self.q - 0.5)
        
        self.B1 = _scalar_constant*np.matmul(y_minus_mu_mi_omega_plus, inverted_part_omega_plus)
        self.B2 = _scalar_constant*np.matmul(y_minus_mu_mi_omega_minus, inverted_part_omega_minus)

        # update sigma_squared
        self.sigma2 = (1/(self.N*self.p))*((TWOPI)**(0.5-self.q))*(np.sum(self.omega_plus_terms) + np.sum(self.omega_minus_terms))

        # optimize latent parameters using Method of Moving Asymptotes
        # self.optimize_means()
        # self.optimize_Sigma()
        print('You are here')

        # TODO: Split up latent mean optimization based on Omega plus, Omega minus

        # optimize latent parameters using gradient ascent
        # TODO: get progress bar working for gradient ascent
        for index, (mi, Si) in enumerate(zip(self.latent_means, self.latent_Sigmas)):
            self.index = index

            loglik_old = self.LL_mean(mi, gradient_mean=np.empty((0, 0)))

            mi_old = mi.flatten()
            mi_new  = mi_old - self.step_size*self.gradient_wrt_latent_mean(mi_old)

            loglik_new = self.LL_mean(mi_new, gradient_mean=np.empty(0))
            
            # update until absolute error is less than tolerance
            while np.abs(loglik_old-loglik_new) > self.tolerance:
                mi_old = mi_new
                loglik_old = loglik_new

                mi_new = mi_old - self.step_size*self.gradient_wrt_latent_mean(mi_old)
                loglik_new = self.LL_mean(mi_new, gradient_mean=np.empty(0))

                # decrease the step size when we're getting close to a solution
                if np.abs(loglik_old-loglik_new) < 2*self.tolerance:
                    self.step_size *= .5
                
            # update the latent means with new optimum
            self.latent_means[index] = mi_new.reshape(-1, 1)

            # now work on the latent Sigmas
            # find out if the index is in either of the Omega plus or Omega minus index sets
            if self.index in self.I1:
                # 
                pass
            elif self.index in self.I2:
                pass
            loglik_old = self.LL_Sigma(Si, gradient_Sigma=np.empty(0))

            Si_old = Si.flatten()
            Si_new = Si_old - self.step_size*self.gradient_wrt_latent_Sigma(Si_old)

            while np.abs(loglik_old-loglik_new) > self.tolerance:
                Si_old = Si_new
                loglik_old = loglik_new

                Si_new = Si_old - self.step_size*self.gradient_wrt_latent_Sigma(Si_old)
                loglik_new = self.LL_Sigma(Si_new, gradient_Sigma=np.empty(0))

            # update latent Sigmas with new optimum
            self.latent_Sigmas[index] = np.diag(np.diag(Si_new.reshape(self.q, self.q))) # this zeros all off-diagonal elements
            # self.latent_Sigmas[index] = Si_new.reshape(self.q, self.q)

    def optimize_model(self):
        """ Perform the EM algorithm over the model parameters B1, B2, and sigma2 
            and latent means and covariance matrices
        """

        # keep track of loglikelihood
        L_new = self.loglikelihood()
        L_old = 0

        self._loglik_results = [L_new]
        count = 0

        # keep performing EM until the change in loglikelihood is less than the tolerance
        print('Optimizing model parameters using the EM algorithm...')
        while np.abs(L_new - L_old) > self.tolerance:
            L_old = L_new
            
            # update the model parameters
            self.M_step()
            count += 1
            print("\r{}".format(count))
            
            L_new = self.loglikelihood()
            self._loglik_results.append(L_new)

        self.loglik = L_new
        self._loglik_results.append(L_new)

        print('Optimal parameters reached.')
        print('Log-likelihood at the optimum: {}'.format(self.loglik))
        
    def get_result(self):
        """ Estimate the positions of the latent variables """

        M_tilde_1 = np.stack([mi.T for mi in self.latent_means if mi[-1] >= 0])
        M_tilde_2 = np.stack([mi.T for mi in self.latent_means if mi[-1] < 0])

        self.W1 = [np.matmul(mi, self.B1.T) for mi in M_tilde_1]
        self.W2 = [np.matmul(mi, self.B2.T) for mi in M_tilde_2]

    def plot_result(self):
        """ Plot the latent means transformed through B1 and B2 """

        ax1 = plt.axes(projection='3d')       

        # extract the individual axis coordinates
        self.x = self.coords[:, 0]
        self.y = self.coords[:, 1]
        self.z = self.coords[:, 2]

        # first plot the latent points in Omega_plus
        ax1.scatter3D(self.x, self.y, self.z, color='red')

        # plot the latent points in Omega_minus
        self.result_x1 = [row[:, 0] for row in self.W1]
        self.result_y1 = [row[:, 1] for row in self.W1]
        self.result_z1 = [row[:, 2] for row in self.W1]

        self.result_x2 = [row[:, 0] for row in self.W2]
        self.result_y2 = [row[:, 1] for row in self.W2]
        self.result_z2 = [row[:, 2] for row in self.W2]
        ax1.scatter3D(self.result_x1, self.result_y1, self.result_z1, color='blue')
        ax1.scatter3D(self.result_x2, self.result_y2, self.result_z2, color='black')

        ax2 = plt.axes(projection='3d')

        plt.show()

if __name__ == '__main__':
    
    p = Problem(n_obs=101, data_type='hinge', latent_dimension=2, tolerance=1e-2)  
    
    print(p)
    p.plot()
    
    # p.optimize_model()
    
    # p.get_result()
    # p.plot_result()