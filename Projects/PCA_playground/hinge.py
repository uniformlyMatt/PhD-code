import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

class Hinge:

    def __init__(self, n_obs, mean, cov):
        """ Create a surface composed of two intersecting planes in 3D.
            The points in the dataset will be Gaussian distributed.
        """
        self.N = n_obs

        # create a 2D Gaussian
        self.obs = np.random.multivariate_normal(mean, cov, self.N)

        # create a bend in the dataset
        self.obs = [np.append(arr=row, values=np.abs(row[0])) for row in self.obs]

        # extract the individual coordinates
        self.x = [item[0] for item in self.obs]
        self.y = [item[1] for item in self.obs]
        self.z = [item[2] for item in self.obs]

        # need this for applying a rotation
        self.coords = np.array([np.array([i, j, k]) for i, j, k in zip(self.x, self.y, self.z)])

    def rotate(self, alpha=0., beta=0., gamma=0.):
        """ Wrapper for the Rotation.from_euler function 

            args:
            -----
            :alpha: float
                rotation about the x axis

            :beta: float
                rotation about the y axis

            :gamma: float
                rotation about the z axis
        """

        self.rotation_matrix = [
            [np.cos(beta)*np.cos(gamma), np.sin(alpha)*np.sin(beta)*np.cos(gamma)-np.cos(alpha)*np.sin(gamma), np.cos(alpha)*np.sin(beta)*np.cos(gamma)+np.sin(alpha)*np.sin(gamma)],
            [np.cos(beta)*np.sin(gamma), np.sin(alpha)*np.sin(beta)*np.cos(gamma)+np.cos(alpha)*np.sin(gamma), np.cos(alpha)*np.sin(beta)*np.cos(gamma)-np.sin(alpha)*np.sin(gamma)],
            [-np.sin(beta), np.sin(alpha)*np.cos(beta), np.cos(alpha)*np.cos(beta)]
        ]

        self.rotation = Rotation.from_matrix(self.rotation_matrix)

        # apply the rotation
        self.coords = np.array([self.rotation.apply(v) for v in self.coords])

        # extract the individual axis coordinates
        self.x = self.coords[:, 0]
        self.y = self.coords[:, 1]
        self.z = self.coords[:, 2]

    def plot(self):
        """ Plot the dataset """
        self.ax = plt.axes(projection='3d')

        self.ax.scatter3D(self.x, self.y, self.z, color='red')
        plt.show()
                

if __name__ == "__main__":
    cov = np.diag(np.random.randn(2)**2)

    y = Hinge(n_obs=81, mean=[0, 0], cov=cov)
    y.rotate(alpha=45., beta=45., gamma=45.)

    y.plot()