import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

class DemoData:

    def __init__(self, kind: str, n_obs: int):
        self.kind = kind
        self.n_obs = n_obs

        assert self.kind in ['example1', 'example2']

        if self.kind == 'example1':
            # create a 2D Gaussian and shift one half down by 4 units
            self.n_dim = 2
            self.q = 2
            self.coords = np.random.normal(0, 1, (self.n_obs, self.q))
            self.intensity = (2*np.pi)**(-self.q/2)*np.exp(-.5*(self.coords[:, 0]**2 + self.coords[:, 1]**2))

            for point in self.coords:
                if point[0] >= 0:
                    point[1] -= 3
            
        if self.kind == 'example2':
            # create a 2D Gaussian and z coords according to 1/x
            self.n_dim = 3
            self.q = 2

            coords = np.random.normal(0, 1, (self.n_obs, self.q))
            self.intensity = (2*np.pi)**(-self.q/2)*np.exp(-.5*(coords[:, 0]**2 + coords[:, 1]**2))

            z = np.array(
                [
                    np.max([np.min([2/x**2, 10]), -10]) for x in coords[:, 0]
                ]
            ).reshape(-1, 1)

            self.coords = np.hstack([coords, z])
            
    def plot(self, ax):
        """ Plot the dataset """
        if self.kind == 'example1':

            self.x = self.coords[:, 0]
            self.y = self.coords[:, 1]
            
            ax.scatter(self.x, self.y, c=self.intensity)
            ax.set_aspect('equal')
        elif self.kind == 'example2':
            ax = plt.axes(projection='3d')

            self.x = self.coords[:, 0]
            self.y = self.coords[:, 1]
            self.z = self.coords[:, 2]

            ax.scatter3D(self.x, self.y, self.z, color='red')

        if __name__ == '__main__':
            plt.show()

if __name__ == '__main__':

    # example1 = DemoData(kind='example1', n_obs=500)
    # plt.scatter(example1.coords[:,0], example1.coords[:,1], c=example1.intensity)
    # plt.show()

    example2 = DemoData(kind='example2', n_obs=500)

    # print(example2.coords.shape)

    ax = plt.axes(projection='3d')
    ax.scatter3D(example2.coords[:,0], example2.coords[:,1], example2.coords[:,2])
    plt.show()