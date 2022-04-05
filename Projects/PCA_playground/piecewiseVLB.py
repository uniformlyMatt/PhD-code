from re import S
import numpy as np
import numpy.linalg as LA
from scipy.special import erf, erfc
from helix import Helix
from config import *
import nlopt
from alive_progress import alive_bar

class Problem:
    def __init__(self, n_obs, latent_dimension=2, timeout=15, tolerance=1e-2):
        self.q = latent_dimension
        self.N = n_obs  # number of observations
        self.timeout = timeout

        # define step size for gradient ascent
        self.step_size = 0.01

        # setup the observations
        self.obs = Helix(radius=5, slope=2, num_points=self.N).coords.T

        self.p = self.obs.shape[0]   # dimension of the observation space
        
        self.tolerance = tolerance   # tolerance for the loglikelihood in the EM algorithm

        # initialize a set of means and standard deviations for the latent variables
        self.latent_means = [np.random.randn(self.q, 1)**2 for _ in range(self.N)]
        self.latent_variances = [np.abs(np.random.randn(self.q))+1 for _ in range(self.N)]
        self.latent_Sigmas = [np.diag(var) for var in self.latent_variances]

        # these are the 's' parameters when nu=e_q, beta=0
        # ss = [-mi[-1]*np.sqrt(si[-1]) for mi, si in zip(latent_means, latent_variances)]

        # set starting values for sigma2, mu, B_1, B_2, g_1, and g_2
        self.mu = np.mean(self.obs, axis=1).reshape(1, -1) # the optimal value for mu is the empirical mean of the data
        self.sigma2 = np.random.rand()       # set to random positive number
        self.B1 = np.random.randn(self.p, self.q)
        self.B2 = np.random.randn(self.p, self.q)

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

    def __str__(self):
        """ Overloading the print function. This presents an informative 
            string display when the object is printed.
        """
        result = "\nPiecewise Linear Probabilistic PCA\n"
        result += '='*(len(result)-2) + '\n\n'

        result += 'Model parameters\n\n'
        result += 'Number of observations: {}\nData dimensions: {}\nLatent dimensions: {}\n'.format(self.N, self.p, self.q)

        return result

    def loglikelihood(self):
        """ Compute the log-likelihood function.
            Compute the long summation terms for the log-likelihood. 
        """
        # update the latent variances based on the updated latent covariances matrices
        self.latent_variances = [np.diag(Si) for Si in self.latent_Sigmas]

        # these are the 's' parameters when nu=e_q, beta=0
        ss = [-mi[-1]*np.sqrt(si[-1]) for mi, si in zip(self.latent_means, self.latent_variances)]

        determinants = [np.prod(np.diag(Sigma)) for Sigma in self.latent_Sigmas]
        traces = [np.trace(Sigma) for Sigma in self.latent_Sigmas]

        miTmi = [sum(mi**2)**2 for mi in self.latent_means]

        a1 = [-tr - mi_norm + np.log(det) for tr, det, mi_norm in zip(traces, determinants, miTmi)]

        B_times_mean = [np.matmul(self.B1+self.B2, mi) for mi in self.latent_means]
        self.a2 = [
            np.matmul(
                yi - self.mu, 
                (yi - self.mu).T
            ).item() - np.matmul(
                yi - self.mu, 
                Bm
            ).item() for yi, Bm in zip(self.Y, B_times_mean)
        ]

        a3_scalars = [SQRT_PI_OVER_2*erfc(si/ROOT2)+si*np.exp(-si**2/2) for si in ss]
        a3 = [
            self.g1*sc*np.trace(
                np.matmul(
                    self.B1.T, 
                    np.matmul(self.B1, Sigma)
                )
            ).item() for Sigma, sc in zip(self.latent_Sigmas, a3_scalars)
        ]

        a4_scalars = [SQRT_PI_OVER_2*(erf(si/ROOT2)+1)-si*np.exp(-si**2/2) for si in ss]
        a4 = np.array([
            self.g2*sc*np.trace(
                np.matmul(
                    self.B2.T, 
                    np.matmul(
                        self.B2, 
                        Sigma
                    )
                )
            ).item() for Sigma, sc in zip(self.latent_Sigmas, a4_scalars)
        ],
        dtype='object'
        )

        a5_inner = [
            erfc(si/ROOT2)*np.matmul(
                self.B1.T, 
                self.B1
            ) + (erf(si/ROOT2)+1)*np.matmul(
                self.B2.T, 
                self.B2
            ) for si in ss]
        a5 = [
            SQRT_PI_OVER_2*np.matmul(mi.T, np.matmul(Bi, mi)) for mi, Bi in zip(self.latent_means, a5_inner)
        ]

        # convert all list of 1d arrays to lists of floats
        self.a1 = [element.item()*0.5 for element in a1]
        self.a3 = [element.item() for element in a3]
        self.a4 = [element.item() for element in a4]
        self.a5 = [element.item() for element in a5]

        scalars = self.N*self.q/2 - self.N*self.p/2*np.log(TWOPI*self.sigma2)

        total = sum(
            [
                item1 - 1/(2*self.sigma2)*item2 + (TWOPI)**(1/2-self.q)*(item3 + item4 + item5) 
                for item1, item2, item3, item4, item5 in zip(self.a1, self.a2, self.a3, self.a4, self.a5)
            ]
        )

        return total + scalars

    def LL_Sigma(self, Sigma, gradient_Sigma):
        """ Optimize the log-likelihood of the variational lower bound wrt  
            the latent covariance matrix Sigma.
        """

        if gradient_Sigma.size > 0:
            gradient_Sigma[:] = self.gradient_wrt_latent_Sigma(Si=Sigma)

        self.latent_Sigmas[self.index] = Sigma.reshape(self.q, self.q)

        loglik = self.loglikelihood()
        return loglik

    def gradient_wrt_latent_Sigma(self, Si):
        """ Computes the gradient of the loglikelihood with 
            respect to a single latent covariance matrix.
        """
        # extract the square root of the q,q element from the covariance matrix while it's still flattened
        s = np.sqrt(Si[-1])       

        # reshape the latent covariance matrix from a flat vector to a matrix
        Si = Si.reshape(self.q, self.q)

        # since Si is a diagonal matrix, we know the inverse is just the reciprocal
        Si_inverse = np.diag(1/np.diag(Si))

        mi = self.latent_means[self.index]
        si = -mi[-1]*s

        scalars = -0.5*(np.eye(self.q) + Si_inverse)

        # derivative of scalar si wrt Sigma_i
        dsdSi = np.zeros((self.q, self.q))
        dsdSi[-1][-1] = -0.5*mi[-1]/s

        # precompute some of the required terms involving B1, B2
        B1TB1 = np.matmul(self.B1.T, self.B1)
        B2TB2 = np.matmul(self.B2.T, self.B2)
        B1TB1_minus_B2TB2 = B1TB1 - B2TB2

        inner_exp_terms = np.exp(-si**2/2)*(
            si*(B1TB1_minus_B2TB2) + \
            dsdSi*(si**2*(np.trace(np.matmul(B2TB2, Si)) - np.trace(np.matmul(B1TB1, Si))) - np.matmul(np.matmul(mi.T, B1TB1_minus_B2TB2), mi))
        )

        inner_erf_terms = SQRT_PI_OVER_2*erfc(si/ROOT2)*B1TB1 + (SQRT_PI_OVER_2*erf(si/ROOT2)+1)*B2TB2

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
        
        # update sigma_squared
        self.sigma2 = 1/(self.N*self.p)*sum([item2 + (TWOPI)**(1/2-self.q)*(item3+item4+item5) for item2, item3, item4, item5 in zip(self.a2, self.a3, self.a4, self.a5)])

        ss = [-mi[-1]*np.sqrt(si[-1]) for mi, si in zip(self.latent_means, self.latent_variances)]
        
        # update the linear transformations B1 and B2
        to_invertB1_part1 = [self.g1*Sigma*(SQRT_PI_OVER_2*erfc(si/ROOT2) + si*np.exp(-si**2/2)) for Sigma, si in zip(self.latent_Sigmas, ss)]
        to_invertB1_part2 = [SQRT_PI_OVER_2*np.matmul(mi, mi.T)*erfc(si/ROOT2) for mi, si in zip(self.latent_means, ss)]
        to_invertB1 = sum([p1 + p2 for p1, p2 in zip(to_invertB1_part1, to_invertB1_part2)])

        to_invertB2_part1 = [self.g1*Sigma*(SQRT_PI_OVER_2*(erf(si/ROOT2)+1) - si*np.exp(-si**2/2)) for Sigma, si in zip(self.latent_Sigmas, ss)]
        to_invertB2_part2 = [SQRT_PI_OVER_2*np.matmul(mi, mi.T)*(erf(si/ROOT2)+1) for mi, si in zip(self.latent_means, ss)]
        to_invertB2 = sum([p1 + p2 for p1, p2 in zip(to_invertB2_part1, to_invertB2_part2)])

        y_minus_mu_mi = sum(
            [
                np.matmul(
                    (yi - self.mu).reshape(-1, 1),
                    mi.T
                ) for yi, mi in zip(self.Y, self.latent_means)
            ]
        )

        inverted_part_B1 = LA.inv(to_invertB1)
        inverted_part_B2 = LA.inv(to_invertB2)
        
        self.B1 = 1/(2*(TWOPI)**(1/2-self.q))*np.matmul(y_minus_mu_mi, inverted_part_B1)
        self.B2 = 1/(2*(TWOPI)**(1/2-self.q))*np.matmul(y_minus_mu_mi, inverted_part_B2)

        # optimize latent parameters using Method of Moving Asymptotes
        # self.optimize_means()
        # self.optimize_Sigma()

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
        """ Perform the EM algorithm over the model parameters B1, B2, and sigma2 """

        L_new = self.loglikelihood()
        L_old = 0

        # keep performing EM until the change in loglikelihood is less than the tolerance
        print('Optimizing model parameters using the EM algorithm...')
        while np.abs(L_new - L_old) > self.tolerance:
            L_old = L_new
            
            # update the model parameters
            self.M_step()
            
            L_new = self.loglikelihood()
        print('Optimal parameters reached.')

if __name__ == '__main__':
    
    p = Problem(n_obs=21, latent_dimension=2, tolerance=1e-2)  
    
    print(p)
    p.optimize_model()
    # p.optimize_means()
    # p.optimize_Sigma()
    
    print(p.latent_Sigmas[1])
    print(p.latent_means[1])