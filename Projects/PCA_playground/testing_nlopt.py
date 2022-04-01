import numpy as np
import numpy.linalg as LA
from scipy.special import erf, erfc
import nlopt
from helix import Helix
from config import *

# setup the data
N = 11
Y = Helix(radius=2, slope=1, num_points=N).coords.T
p = 3
q = 2

# initialize a set of means and standard deviations for the latent variables
latent_means = [np.random.randn(q, 1) for _ in range(N)]
latent_variances = [np.random.randn(q)**2 for _ in range(N)]
latent_Sigmas = [np.diag(var) for var in latent_variances]

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

def compute_gradient(Y, mi, latent_Sigmas, B1, B2, ss, mu, g1, g2, sigma2, index):
    """ Compute the derivative of the loglikelihood wrt a single latent mean. """

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

    # qth_terms = [-var[-1]*eq*(item1 + item2 + item3) for var, item1, item2, item3 in zip(latent_variances, b1, b2, b3)]

    # return sum([outer_term + (TWOPI)**(1/2-q)/(2*sigma2)*inner_term for outer_term, inner_term in zip(b0, inner_terms)])

    result = b0 + b4

    # update the qth element with the corresponding derivative elements
    result[-1] += (TWOPI)**(1/2-q)/(2*sigma2)*(b1 + b2 + b3)

    return result.flatten()

def loglikelihood(mean, grad):
    """ Compute the log-likehood of the variational lower bound for the problem. 
        This comprises the E-step in the EM algorithm.
    """

    # update the global latent_means list
    latent_means[index] = mean

    if grad.size > 0:
        # update the gradient
        grad[:] = compute_gradient(
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

# def myfunc(x, grad):
#     if grad.size > 0:
#         grad[0] = 0.0  
#         grad[1] = 0.5 / np.sqrt(x[1])
#     return np.sqrt(x[1])

# def myconstraint(x, grad, a, b):
#     if grad.size > 0:
#         grad[0] = 3 * a * (a*x[0] + b)**2
#         grad[1] = -1.0
#     return (a*x[0] + b)**3 - x[1]

opt = nlopt.opt(nlopt.LD_MMA, q)

opt.set_lower_bounds([-float('inf')]*q)
opt.set_upper_bounds([float('inf')]*q)
opt.set_max_objective(loglikelihood)

# opt.add_inequality_constraint(lambda x, grad: myconstraint(x,grad,2,0), 1e-8)
# opt.add_inequality_constraint(lambda x, grad: myconstraint(x,grad,-1,1), 1e-8)

opt.set_xtol_rel(1e-4)

x = []

for index in range(N):
    print('Optimizing for latent mean number {}'.format(index))
    print('-------------------------------------\n')
    mi = latent_means[index].flatten()
    xi = opt.optimize(mi)

    x.append(xi)
    
    # update the latent means with the new optimum
    print(mi)
    latent_means[index] = xi.reshape(-1, 1)
    print('Maximum value: {}'.format(opt.last_optimum_value()))
    print('Optimum at {}\n'.format(xi))
# print("optimum at ", *x)
# print("minimum value = ", minf)
# print("result code = ", opt.last_optimize_result())