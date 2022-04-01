import numpy as np
import numpy.linalg as LA
from scipy.special import erf, erfc
from helix import Helix
from config import *
# import nlopt

class VLB:
    """ Implements the variational lower bound for the piecewise 
        probabilistic PCA model.
    """
    def __init__(self, obs: np.ndarray, q: int, tolerance=0.01):
        """ 
        :obs: np.ndarray
            matrix of observations; each column is one p-dimensional observation
        :p: int
            dimension of the observation space 
        :q: int
            dimension of the latent space
        """
        self.obs = obs
        self.N = self.obs.shape[-1]  # number of columns in obs
        self.p = self.obs.shape[0]   # number of observed random variables is the number of rows
        self.q = q                   # dimension of the latent space

        self.tolerance = tolerance   # tolerance for the loglikelihood in the EM algorithm

        # initialize a set of means and standard deviations for the latent variables
        self.latent_means = [np.random.randn(q, 1) for _ in range(self.N)]
        self.latent_variances = [np.random.randn(q)**2 for _ in range(self.N)]
        self.latent_Sigmas = [np.diag(var) for var in self.latent_variances]

        # these are the 's' parameters when nu=e_q, beta=0
        self.ss = [-mi[-1]*si[-1] for mi, si in zip(self.latent_means, self.latent_variances)]

        # set starting values for sigma2, mu, B_1, B_2, g_1, and g_2
        self.mu = np.mean(self.obs, axis=1).reshape(1, -1) # the optimal value for mu is the empirical mean of the data
        self.sigma2 = np.random.rand()       # set to random positive number
        self.B_1 = np.random.randn(self.p, self.q)
        self.B_2 = np.random.randn(self.p, self.q)

        self.g_1 = np.random.rand()
        self.g_2 = np.random.rand()

        # I want the observations to be 1xp arrays for later computations
        self.Y = [yi.reshape(1, -1) for yi in self.obs.T]

    def __str__(self):
        """ Display problem setup """
        result = '\nPiecewise Probabilistic PCA\n'
        result += '='*(len(result)-2) + '\n'

        result += 'Number of obs: {}\nData dimension: {}\nLatent dimension: {}\n'.format(self.N, self.p, self.q)
        result += 'B1:\n' + str(self.B_1) + '\n'
        result += 'B2:\n' + str(self.B_2) + '\n'

        return result

    def compute_terms(self):
        """ Compute the summation terms for the log-likelihood. """
        a1 = [-np.trace(Sigma) - np.matmul(mi.T, mi) + np.log(LA.det(Sigma)) for Sigma, mi in zip(self.latent_Sigmas, self.latent_means)]

        B_times_mean = [np.matmul(self.B_1+self.B_2, mn) for mn in self.latent_means]
        self.a2 = [
            np.matmul(
                yi - self.mu, 
                (yi - self.mu).T
            ).item() - np.matmul(
                yi - self.mu, 
                Bm
            ).item() for yi, Bm in zip(self.Y, B_times_mean)
        ]

        a3_scalars = [SQRT_PI_OVER_2*erfc(si/ROOT2)+si*np.exp(-si**2/2) for si in self.ss]
        a3 = [
            self.g_1*sc*np.trace(
                np.matmul(
                    self.B_1.T, 
                    np.matmul(self.B_1, Sigma)
                )
            ).item() for Sigma, sc in zip(self.latent_Sigmas, a3_scalars)
        ]

        a4_scalars = [SQRT_PI_OVER_2*(erf(si/ROOT2)+1)-si*np.exp(-si**2/2) for si in self.ss]
        a4 = np.array([
            self.g_2*sc*np.trace(
                np.matmul(
                    self.B_2.T, 
                    np.matmul(
                        self.B_2, 
                        Sigma
                    )
                )
            ).item() for Sigma, sc in zip(self.latent_Sigmas, a4_scalars)
        ])

        a5_inner = [
            erfc(si/ROOT2)*np.matmul(
                self.B_1.T, 
                self.B_1
            ) + (erf(si/ROOT2)+1)*np.matmul(
                self.B_2.T, 
                self.B_2
            ) for si in self.ss]
        a5 = [
            SQRT_PI_OVER_2*np.matmul(mi.T, np.matmul(Bi, mi)) for mi, Bi in zip(self.latent_means, a5_inner)
        ]

        # convert all list of 1d arrays to lists of floats
        self.a1 = [element.item()*0.5 for element in a1]
        self.a3 = [element.item() for element in a3]
        self.a4 = [element.item() for element in a4]
        self.a5 = [element.item() for element in a5]

    def loglikelihood(self):
        """ Compute the log-likehood of the variational lower bound for the problem. 
            This comprises the E-step in the EM algorithm.
        """
        self.compute_terms()

        scalars = self.N*self.q/2 - self.N*self.p/2*np.log(TWOPI*self.sigma2)

        total = sum(
            [
                item1 - 1/(2*self.sigma2)*item2 + (TWOPI)**(1/2-self.q)*(item3 + item4 + item5) 
                for item1, item2, item3, item4, item5 in zip(self.a1, self.a2, self.a3, self.a4, self.a5)
            ]
        )

        return total + scalars

    def update(self):
        """ Update the model parameters based on the maximum likelihood estimates 
            for the parameters.

            This comprises the M-step of the EM algorithm.
        """
        
        # update sigma_squared
        self.sigma2 = 1/(self.N*self.p)*sum([item2 + (TWOPI)**(1/2-self.q)*(item3+item4+item5) for item2, item3, item4, item5 in zip(self.a2, self.a3, self.a4, self.a5)])

        # update the linear transformations
        to_invert1_part1 = [self.g_1*Sigma*(SQRT_PI_OVER_2*erfc(si/ROOT2) + si*np.exp(-si**2/2)) for Sigma, si in zip(self.latent_Sigmas, self.ss)]
        to_invert1_part2 = [SQRT_PI_OVER_2*np.matmul(mi, mi.T)*erfc(si/ROOT2) for mi, si in zip(self.latent_means, self.ss)]
        to_invert1 = sum([p1 + p2 for p1, p2 in zip(to_invert1_part1, to_invert1_part2)])

        y_minus_mu_mi = sum(
            [
                np.matmul(
                    (yi - self.mu).reshape(-1, 1),
                    mi.T
                ) for yi, mi in zip(self.Y, self.latent_means)
            ]
        )
        
        self.B_1 = 1/(2*(TWOPI)**(1/2-self.q))*np.matmul(LA.inv(to_invert1), y_minus_mu_mi)

    def optimize(self):
        """ Perform the EM algorithm until loglikehood error tolerance is reached. """

        L = self.loglikelihood()
        L_new = 0

        # while np.abs(L - L_new) > self.tolerance:
        #     self.update()
        #     L_new = self.loglikelihood()


if __name__ == '__main__':
    h = Helix(radius=2, slope=1, num_points=10)

    pr = VLB(obs=h.coords.T, q=2)
    pr.loglikelihood()
    pr.update()