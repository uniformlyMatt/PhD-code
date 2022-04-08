import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

class Helix:
    """ Produces a noisy helix in 3 dimensional space. """

    def __init__(self, radius: float, slope: float, num_points: int, variance=1.):
        self.radius = radius
        self.slope = slope
        self.num_points = num_points
        self.variance = variance

        # the points along the helix are distributed uniformly along the helix, 
        # but Gaussian in the cross-section (only x and y need to be Gaussian distributed)
        self.t = np.linspace(0, 6*np.pi, num_points)
        self.x = radius*np.cos(self.t) + np.random.normal(0, scale=self.variance, size=num_points)
        self.y = radius*np.sin(self.t) + np.random.normal(0, scale=self.variance, size=num_points)
        self.z = slope*self.t

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
        """ Plot the helix """
        self.ax = plt.axes(projection='3d')

        self.ax.scatter3D(self.x, self.y, self.z, color='red')
        plt.show()

# h1 = Helix(radius=5, slope=1, num_points=101, variance=0.75)
# h2 = Helix(radius=1.2, slope=1, num_points=101, variance=0.98)

# h1.rotate(alpha=12, beta=0, gamma=5)

# ax = plt.axes(projection='3d')

# ax.scatter3D(h1.x, h1.y, h1.z, color='red')
# ax.scatter3D(h2.x, h2.y, h2.z, color='blue')

# plt.show()