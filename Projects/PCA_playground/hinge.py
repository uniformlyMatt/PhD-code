import numpy as np
import matplotlib.pyplot as plt

class Hinge:

    def __init__(self, n_obs, mean, cov):
        """ Create a surface composed of two intersecting planes in 3D.
            The points in the dataset will be Gaussian distributed.
        """
        self.N = n_obs

        # create a 2D Gaussian
        self.Y = np.random.multivariate_normal(mean, cov, self.N)

    def plot(self):
        """ Plot the dataset """
        self.ax = plt.axes(projection='3d')

        x = [item[0] for item in self.Y]
        y = [item[1] for item in self.Y]
        z = [0 for _ in self.Y]

        self.ax.scatter3D(x, y, z, color='red')
        plt.show()
                

if __name__ == "__main__":
    cov = np.diag(np.random.randn(2)**2)

    y = Hinge(n_obs=81, mean=[0, 0], cov=cov)

    print(y.Y[:10])

    y.plot()