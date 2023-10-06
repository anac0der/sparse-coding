from ksvd import ApproximateKSVD
from utils import Patches
import numpy as np
from scipy.stats import chi2
from utils import create_hermite2d_vander


def ksvd_denoising(noisy_image, patch_size, n_components, scale, rate, max_iter, L=0, seed=None):
    patches = Patches(noisy_image)

    patch_arr = patches.create_patches(patch_size, reshape=True).astype('float32')
    train_indexes_rate = [i for i in range(patch_arr.shape[0]) if i % rate == 0 and (i // (noisy_image.shape[1] - patch_size + 1)) % rate == 0]
    train_patches_set_rate = patch_arr[train_indexes_rate, :]

    rng = np.random.default_rng(seed=seed)
    D_init = rng.choice(patch_arr, size=n_components, axis=0, replace=False)
    # D_init = create_hermite2d_vander(15, 15, (patch_size, patch_size)).T
    tol = chi2.ppf(0.93, patch_size ** 2) * scale ** 2
    ksvd_rate = ApproximateKSVD(n_components=n_components, max_iter=max_iter, omp_tol=tol)
    D_rate = ksvd_rate.fit(train_patches_set_rate, is_D_init=True, D_init=D_init).components_
    ksvd = ApproximateKSVD(n_components=n_components, max_iter=1, omp_tol=tol)
    D = ksvd.fit(patch_arr, is_D_init=True, D_init=D_rate).components_
    print('Dictionary for all patches is fitted!')
    
    reconstructed_patches = ksvd.gamma_.dot(D)
    reconstructed_patches = reconstructed_patches.reshape((-1, patch_size, patch_size))
    final_image = patches.reconstruct_avg(reconstructed_patches).astype('float32')
    final_image[final_image < 0] = 0
    final_image[final_image > 255] = 255
    if L == 'auto':
        rmse = np.sqrt(np.mean((noisy_image - final_image)**2))
        L = max(rmse / scale - 1, 0)
    elif not isinstance(L, float) and not isinstance(L, int):
        L = 0
    final_image = (L * noisy_image + final_image) / (L + 1)
    return final_image  