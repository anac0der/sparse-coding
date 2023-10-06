import argparse
import cv2
import numpy as np
from denoising import ksvd_denoising
from skimage.restoration import denoise_nl_means


parser = argparse.ArgumentParser()
parser.add_argument('image_path', type=str)
args = parser.parse_args()

image = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE).astype('float32')
# cv2.imwrite(args.image_path, image)

#Algorithm parameters
scale = 50
patch_size = 8
rate = 4
n_components = 256
max_iter = 16
L = 1 / (2 * scale)
#Adding noise to image, you can skip that part
rng = np.random.default_rng(seed=2002)
noise = rng.normal(size=image.shape, scale=scale).astype('float32')
noisy_image = image + noise
# noisy_image = 255 * random_noise(image / 255, mode='s&p', amount=0.1)
noisy_image[noisy_image < 0] = 0
noisy_image[noisy_image > 255] = 255
cv2.imwrite('noisy.png', noisy_image)

final_image = ksvd_denoising(noisy_image, patch_size=patch_size, \
                              n_components=n_components, scale=scale, rate=rate, max_iter=max_iter, \
                                seed=2002, L=L)
# final_image = denoise_nl_means(noisy_image, sigma=scale, h=scale/2)
cv2.imwrite('result.png', final_image)
print(f'Initial PSNR: {(cv2.PSNR(image, noisy_image)):.2f}')
print(f'PSNR after denoising: {(cv2.PSNR(image, final_image)):.2f}')
print(f'RMSE after denoising: {(np.sqrt(np.mean((image - final_image)**2))):.2f}')