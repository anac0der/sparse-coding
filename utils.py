import numpy as np
from numpy.polynomial.hermite import hermvander2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d

def create_hermite2d_vander(x_f, y_f, shape):
    grid_x = np.linspace(-np.sqrt(2 * x_f + 1), np.sqrt(2 * y_f + 1), shape[1])
    grid_y = np.linspace(-np.sqrt(2 * x_f + 1), np.sqrt(2 * y_f + 1), shape[0])
    x, y = np.meshgrid(grid_x, grid_y)

    D = hermvander2d(x, y, [x_f, y_f])
    D *= np.expand_dims(np.exp((-x**2 - y**2) / 2), axis=-1)
    D = D.reshape((-1, D.shape[-1]))
    D /= np.linalg.norm(D, axis=0)

    return D

class Patches:
    def __init__(self, image):
        self.image = image

    def create_patches(self, patch_size, reshape=False):
        patches = np.zeros(((self.image.shape[0] - patch_size + 1) * (self.image.shape[1] - patch_size + 1), patch_size, patch_size))
        k = 0
        for i in range(self.image.shape[0] - patch_size + 1):
            for j in range(self.image.shape[1] - patch_size + 1):
                patches[k, :, :] = self.image[i:i + patch_size, j:j + patch_size]
                k += 1

        if reshape:
            return patches.reshape((patches.shape[0], -1))
        else:
            return patches
    
    def reconstruct_avg(self, patches):
        return reconstruct_from_patches_2d(patches, self.image.shape) 