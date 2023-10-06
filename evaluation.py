from denoising import ksvd_denoising
import os
import cv2
import numpy as np
from skimage.restoration import denoise_nl_means

data_folder = './test_data'

#Algorithm parameters
patch_size = 8
rate = 4
n_components = 256
max_iter = 16
n_sim = 5

results = dict()
rng = np.random.default_rng(seed=2002)

for scale in [5, 10, 15, 20, 25, 50]:
    L = 1 / (2 * scale)
    for root, dirs, images in os.walk(data_folder):
        for image in images:
            img_arr = cv2.imread(os.path.join(root, image), cv2.IMREAD_GRAYSCALE).astype('float32')
            name = image.split('.')[0]
            results[name] = {'PSNR': [], 'RMSE': []}
            print(f'Denoising {image}...')
            for i in range(n_sim):
                noise = rng.normal(size=img_arr.shape, scale=scale).astype('float32')
                noisy_image = img_arr + noise
                noisy_image[noisy_image < 0] = 0
                noisy_image[noisy_image > 255] = 255

                # final_image = ksvd_denoising(noisy_image, patch_size=patch_size, \
                                # n_components=n_components, scale=scale, rate=rate, max_iter=max_iter, \
                                    # seed=2002, L=L).astype('float32')
                final_image = denoise_nl_means(noisy_image, sigma=scale, h=scale/2)
                results[name]['PSNR'].append(cv2.PSNR(img_arr, final_image))
                results[name]['RMSE'].append(np.sqrt(np.mean((img_arr - final_image)**2)))

    # fname = f'nlm_exp_051023_scale={scale}_patch={patch_size}_rate={rate}_iter={max_iter}_L={(L):.3f}_maxiterfull=1.txt'
    fname = f'051023_nlm_scale={scale}_h={scale/2}'
    with open(fname, 'w') as f:
        f.write(f'Denoising results (noise level = {scale})\n')
        f.write('\n')
        for image in results.keys():
            f.write(f'--- {image} ---\n')
            f.write(f'Mean PSNR: {(sum(results[image]["PSNR"]) / n_sim):.2f}, by simulation: {results[image]["PSNR"]}\n') 
            f.write(f'Mean RMSE: {(sum(results[image]["RMSE"]) / n_sim):.2f}, by simulation: {results[image]["RMSE"]}\n')    
            f.write('\n')
    print(f'Scale {scale} is done!')