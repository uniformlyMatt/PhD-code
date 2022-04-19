import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

class DemoData:

    def __init__(self, kind: str, n_obs: int, mean: np.array, covariance: float, alpha=15., beta=15., gamma=15.):
        self.kind = kind
        self.n_obs = n_obs
        self.mean = mean
        self.covariance = covariance

        # construct the rotation matrix
        # self.rotation_matrix = [
        #     [np.cos(beta)*np.cos(gamma), np.sin(alpha)*np.sin(beta)*np.cos(gamma)-np.cos(alpha)*np.sin(gamma), np.cos(alpha)*np.sin(beta)*np.cos(gamma)+np.sin(alpha)*np.sin(gamma)],
        #     [np.cos(beta)*np.sin(gamma), np.sin(alpha)*np.sin(beta)*np.cos(gamma)+np.cos(alpha)*np.sin(gamma), np.cos(alpha)*np.sin(beta)*np.cos(gamma)-np.sin(alpha)*np.sin(gamma)],
        #     [-np.sin(beta), np.sin(alpha)*np.cos(beta), np.cos(alpha)*np.cos(beta)]
        # ]
        # self.rotation_matrix = [
        #     [np.cos(beta)*np.cos(alpha), np.cos(beta)*np.sin(alpha), np.sin(beta)],
        #     [-np.cos(gamma)*np.sin(alpha), np.cos(gamma)*np.cos(alpha), 0],
        #     [np.sin(gamma)*np.sin(alpha)-np.cos(gamma)*np.sin(beta)*np.cos(alpha), -np.sin(gamma)*np.cos(alpha)-np.cos(gamma)*np.sin(beta)*np.sin(alpha), np.cos(gamma)*np.cos(beta)]
        # ]
        # self.rotation_vector = np.random.rand(3)
        self.rotation_vector = np.random.randint(low=0, high=3, size=3)
        self.axis = self.rotation_vector/np.linalg.norm(self.rotation_vector)

        # self.rotation = Rotation.from_matrix(self.rotation_matrix)
        self.rotation = Rotation.from_rotvec(beta*self.axis)

        assert self.kind in ['hinge', 'helix', 'mixture']

        if self.kind == 'hinge':
            # create a 2D Gaussian
            self.obs = np.random.multivariate_normal(self.mean, self.covariance, self.n_obs)

            # create a bend in the dataset
            self.obs = [np.append(arr=row, values=np.abs(row[0])) for row in self.obs]

            # extract the individual coordinates
            self.x = [item[0] for item in self.obs]
            self.y = [item[1] for item in self.obs]
            self.z = [item[2] for item in self.obs]

            # need this for applying a rotation
            self.coords = np.array([np.array([i, j, k]) for i, j, k in zip(self.x, self.y, self.z)])

        if self.kind == 'helix':
            # TODO: make these customizable
            radius = 5
            slope = 1

            var = np.random.rand()

            # the points along the helix are distributed uniformly along the helix, 
            # but Gaussian in the cross-section (only x and y need to be Gaussian distributed)
            self.t = np.linspace(-3*np.pi, 3*np.pi, self.n_obs)
            self.x = radius*np.cos(self.t) + np.random.normal(0, scale=var, size=self.n_obs)
            self.y = radius*np.sin(self.t) + np.random.normal(0, scale=var, size=self.n_obs)
            self.z = slope*self.t + np.random.rand(self.n_obs)

            # need this for applying a rotation
            self.coords = np.array([np.array([i, j, k]) for i, j, k in zip(self.x, self.y, self.z)])

        if self.kind == 'mixture':
            # create two 2D Gaussian
            self.obs1 = np.random.multivariate_normal(self.mean, self.covariance, self.n_obs)
            self.obs2 = np.random.multivariate_normal(self.mean, self.covariance, self.n_obs)

            self.obs2 *= -1

            z1 = 2*np.random.rand()
            z2 = -z1

            self.obs = [np.append(arr=row, values=z1) for row in self.obs1]
            self.obs += [np.append(arr=row, values=z2) for row in self.obs2]

            # extract the individual coordinates
            self.x = [item[0] for item in self.obs]
            self.y = [item[1] for item in self.obs]
            self.z = [item[2] for item in self.obs]

            # need this for applying a rotation
            self.coords = np.array([np.array([i, j, k]) for i, j, k in zip(self.x, self.y, self.z)])

    def rotate(self):
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

if __name__ == '__main__':
    cov = np.diag(np.random.randn(2)**2)

    helix = DemoData(kind='helix', n_obs=101, mean=None, covariance=1.1)
    hinge = DemoData(kind='hinge', n_obs=101, mean=[0, 0], covariance=cov)
    planes = DemoData(kind='mixture', n_obs=101, mean=[1, 0], covariance=cov)

    # helix.rotate()
    # hinge.rotate()
    planes.rotate()

    # helix.plot()
    # hinge.plot()
    planes.plot()