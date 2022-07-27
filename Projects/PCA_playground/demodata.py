import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

class DemoData:

    def __init__(self, kind: str, n_obs: int):
        self.kind = kind
        self.n_obs = n_obs

        assert self.kind in ['example1', 'example2']

        self.n_dim = 3
        self.q = 2
        coords = np.random.normal(0, 1, (self.n_obs, self.q))
        self.intensity = (2*np.pi)**(-self.q/2)*np.exp(-.5*(coords[:, 0]**2 + coords[:, 1]**2))

        if self.kind == 'example1':
            # create a 2D Gaussian and shift one half down by 4 units
            z = np.array(
                [np.abs(x) + np.random.normal(loc=0., scale=.1) for x in coords[:, 0]]
            ).reshape(-1, 1)

        elif self.kind == 'example2':
            # create a 2D Gaussian and z coords according to 1/x
            z = np.array(
                [np.max([np.min([2/x, 10]), -10]) + np.random.normal(loc=0., scale=.05) for x in coords[:, 1]]
            ).reshape(-1, 1)

        self.coords = np.hstack([coords, z])
        self.X = coords
        self.y = z
            
    def plot(self):
        """ Plot the dataset """
        ax = plt.axes(projection='3d')
        
        self.x = self.coords[:, 0]
        self.y = self.coords[:, 1]
        self.z = self.coords[:, 2]

        ax.scatter3D(self.x, self.y, self.z, c=self.intensity)
        plt.show()

if __name__ == '__main__':
    example = DemoData(kind='example2', n_obs=500)
    example.plot()

    # print(example2.coords.shape)

    