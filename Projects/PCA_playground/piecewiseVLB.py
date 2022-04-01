import numpy as np
import numpy.linalg as LA
from scipy.special import erf, erfc
from helix import Helix
from config import *
import nlopt

def gradient_wrt_latent_mean(Y, mi, latent_Sigmas, B1, B2, ss, mu, g1, g2, sigma2, index):
    """ Computes the gradient of the loglikelihood with 
        respect to a single latent mean.
    """
    # these are the 's' parameters when nu=e_q, beta=0
    si = -mi[-1]*ss[index][-1]
    Sigma = latent_Sigmas[index]
    yi = Y[index]

    B1_plus_B2 = B1 + B2

    b0 = -mi.reshape(-1, 1) + 1/(2*sigma2)*np.matmul(B1_plus_B2.T, (yi-mu).T)

    b1 = (SQRT_PI_OVER_2-1-si**2)*np.exp(-si**2/2)*g1*np.trace(
        np.matmul(
            np.matmul(
                B1.T, B1
            ),
            Sigma
        )
    )

    b2 = si*np.exp(-si**2/2)*g2*np.trace(
        np.matmul(
            np.matmul(
                B2.T,
                B2
            ),
            Sigma
        )
    )

    B1TB1 = np.matmul(B1.T, B1)
    B2TB2 = np.matmul(B2.T, B2)

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
    result[-1] += (TWOPI)**(1/2-q)/(2*sigma2)*(b1 + b2 + b3)

    return result.flatten()

def compute_terms(Y, latent_means, latent_Sigmas, B1, B2, mu, g1, g2):
    """ Helper function for the log-likelihood function.
        Compute the long summation terms for the log-likelihood. 
    """
    # these are the 's' parameters when nu=e_q, beta=0
    ss = [-mi[-1]*si[-1] for mi, si in zip(latent_means, latent_variances)]

    a1 = [-np.trace(Sigma) - np.matmul(mi.T, mi) + np.log(LA.det(Sigma)) for Sigma, mi in zip(latent_Sigmas, latent_means)]

    B_times_mean = [np.matmul(B1+B2, mi) for mi in latent_means]
    a2 = [
        np.matmul(
            yi - mu, 
            (yi - mu).T
        ).item() - np.matmul(
            yi - mu, 
            Bm
        ).item() for yi, Bm in zip(Y, B_times_mean)
    ]

    a3_scalars = [SQRT_PI_OVER_2*erfc(si/ROOT2)+si*np.exp(-si**2/2) for si in ss]
    a3 = [
        g1*sc*np.trace(
            np.matmul(
                B1.T, 
                np.matmul(B1, Sigma)
            )
        ).item() for Sigma, sc in zip(latent_Sigmas, a3_scalars)
    ]

    a4_scalars = [SQRT_PI_OVER_2*(erf(si/ROOT2)+1)-si*np.exp(-si**2/2) for si in ss]
    a4 = np.array([
        g2*sc*np.trace(
            np.matmul(
                B2.T, 
                np.matmul(
                    B2, 
                    Sigma
                )
            )
        ).item() for Sigma, sc in zip(latent_Sigmas, a4_scalars)
    ],
    dtype='object'
    )

    a5_inner = [
        erfc(si/ROOT2)*np.matmul(
            B1.T, 
            B1
        ) + (erf(si/ROOT2)+1)*np.matmul(
            B2.T, 
            B2
        ) for si in ss]
    a5 = [
        SQRT_PI_OVER_2*np.matmul(mi.T, np.matmul(Bi, mi)) for mi, Bi in zip(latent_means, a5_inner)
    ]

    # convert all list of 1d arrays to lists of floats
    a1 = [element.item()*0.5 for element in a1]
    a3 = [element.item() for element in a3]
    a4 = [element.item() for element in a4]
    a5 = [element.item() for element in a5]

    return a1, a2, a3, a4, a5

def loglikelihood(mean, gradient_mean):
    """ Compute the log-likehood of the variational lower bound for the problem. 
        This comprises the E-step in the EM algorithm.

        :mean: np.array
            contains the current value of the latent mean with respect to which
            we are optimizing

        :gradient_mean: np.array
            contains the gradient of the loglikelihood
            with respect to a single latent mean
    """

    # update the global latent_means list
    latent_means[index] = mean

    if gradient_mean.size > 0:
        # update the gradient
        gradient_mean[:] = gradient_wrt_latent_mean(
            Y=Y, 
            mi=mean, 
            latent_Sigmas=latent_Sigmas,
            B1=B1,
            B2=B2,
            ss=ss,
            mu=mu,
            g1=g1,
            g2=g2,
            sigma2=sigma2,
            index=index
        )

    a1, a2, a3, a4, a5 = compute_terms(
        Y=Y, 
        latent_means=latent_means, 
        latent_Sigmas=latent_Sigmas, 
        B1=B1, 
        B2=B2, 
        mu=mu, 
        g1=g1, 
        g2=g2
    )

    scalars = N*q/2 - N*p/2*np.log(TWOPI*sigma2)

    total = sum(
        [
            item1 - 1/(2*sigma2)*item2 + (TWOPI)**(1/2-q)*(item3 + item4 + item5) 
            for item1, item2, item3, item4, item5 in zip(a1, a2, a3, a4, a5)
        ]
    )

    return total + scalars

# def update(self):
#     """ Update the model parameters based on the maximum likelihood estimates 
#         for the parameters.

#         This comprises the M-step of the EM algorithm.
#     """
    
#     # update sigma_squared
#     self.sigma2 = 1/(self.N*self.p)*sum([item2 + (TWOPI)**(1/2-self.q)*(item3+item4+item5) for item2, item3, item4, item5 in zip(self.a2, self.a3, self.a4, self.a5)])

#     # update the linear transformations
#     to_invert1_part1 = [self.g_1*Sigma*(SQRT_PI_OVER_2*erfc(si/ROOT2) + si*np.exp(-si**2/2)) for Sigma, si in zip(self.latent_Sigmas, self.ss)]
#     to_invert1_part2 = [SQRT_PI_OVER_2*np.matmul(mi, mi.T)*erfc(si/ROOT2) for mi, si in zip(self.latent_means, self.ss)]
#     to_invert1 = sum([p1 + p2 for p1, p2 in zip(to_invert1_part1, to_invert1_part2)])

#     y_minus_mu_mi = sum(
#         [
#             np.matmul(
#                 (yi - self.mu).reshape(-1, 1),
#                 mi.T
#             ) for yi, mi in zip(self.Y, self.latent_means)
#         ]
#     )
    
#     self.B_1 = 1/(2*(TWOPI)**(1/2-self.q))*np.matmul(LA.inv(to_invert1), y_minus_mu_mi)

if __name__ == '__main__':
    
    N = 11  # number of observations

    # setup the observations
    Y = Helix(radius=2, slope=1, num_points=N).coords.T

    p = Y.shape[0]   # dimension of the observation space
    q = 2            # dimension of the latent space

    tolerance = 1e-4   # tolerance for the loglikelihood in the EM algorithm

    # initialize a set of means and standard deviations for the latent variables
    latent_means = [np.random.randn(q, 1) for _ in range(N)]
    latent_variances = [np.random.randn(q)**2 for _ in range(N)]
    latent_Sigmas = [np.diag(var) for var in latent_variances]

    # these are the 's' parameters when nu=e_q, beta=0
    ss = [-mi[-1]*si[-1] for mi, si in zip(latent_means, latent_variances)]

    # set starting values for sigma2, mu, B_1, B_2, g_1, and g_2
    mu = np.mean(Y, axis=1).reshape(1, -1) # the optimal value for mu is the empirical mean of the data
    sigma2 = np.random.rand()       # set to random positive number
    B1 = np.random.randn(p, q)
    B2 = np.random.randn(p, q)

    g1 = np.random.rand()
    g2 = np.random.rand()

    # I want the observations to be 1xp arrays for later computations
    Y = [yi.reshape(1, -1) for yi in Y.T]

    # setup the nonlinear optimizers for the nonlinear gradients
    opt_means = nlopt.opt(nlopt.LD_MMA, q)  # MMA is method of moving asymptotes

    # this sets the lower and upper bounds for the unconstrained optimization of the latent means
    opt_means.set_lower_bounds([-float('inf')]*q)
    opt_means.set_upper_bounds([float('inf')]*q)

    # set the error tolerance of the optimizer
    opt_means.set_xtol_rel(1e-4)

    opt_means.set_max_objective(loglikelihood)

    # optimize wrt the latent means
    for index in range(N):
        # TODO: progress bar output instead of printouts
        print('Optimizing for latent mean number {}'.format(index))
        print('-------------------------------------\n')
        mi = latent_means[index].flatten()
        xi = opt_means.optimize(mi)
        
        # update the latent means with the new optimum
        print(mi)
        latent_means[index] = xi.reshape(-1, 1)
        print('Maximum value: {}'.format(opt_means.last_optimum_value()))
        print('Optimum at {}\n'.format(xi))    