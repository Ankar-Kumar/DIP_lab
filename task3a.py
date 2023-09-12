import numpy as np
import cv2
import matplotlib.pyplot as plt
image= cv2.imread('imgg/cameraman.jpg', cv2.IMREAD_GRAYSCALE)
image=cv2.resize(image,(512,512))
height=512
width=512
def addNoise(image):
    noise_image=image.copy()
    noise=0.02
    height,width=512,512

    for h in range(height):
        for w in range(width):
            random_val=np.random.rand()
            if random_val<noise/2:
                noise_image[h,w]=0
            elif random_val<noise:
                noise_image[h,w]=255
    return noise_image
noise_image=addNoise(image)
plt.subplot(2,2,1)
plt.imshow(image,cmap='gray')
plt.subplot(2,2,2)
plt.imshow(noise_image,cmap='gray')
def masking(noise_image,height,width):
    n=3 
    mask = np.ones((n,n), dtype=np.float32) / (n*n*1.0)
    pad_height=n//2
    pad_width=n//2
    spatial_image=np.zeros((height,width),dtype=np.uint8)
    pad_image=np.pad(noise_image,((pad_height,pad_height),(pad_width,pad_width)),mode='constant')

    for h in range(height):
        for w in range(width):
            roi=pad_image[h:h+n,w:w+n]
            weight=np.sum(roi*mask)
            spatial_image[h,w]=weight

    return spatial_image

def masking2(noise_image,height,width):
    n=3
    mask = np.ones((n,n))
    pad_height=n//2
    pad_width=n//2
    spatial_image=np.zeros_like(noise_image)
    pad_image=np.pad(noise_image,((pad_height,pad_height),(pad_width,pad_width)),mode='constant')

    for h in range(height):
        for w in range(width):
            roi=pad_image[h:h+n,w:w+n]
            weight=np.median(roi*mask)
            spatial_image[h,w]=weight

    return spatial_image

def PSNR(original,noisy):
   mse = np.mean((original.astype(np.float32) - noisy.astype(np.float32)) ** 2)
   max_pixel_value = 255.0
   psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
   return psnr


spatial_image=masking(noise_image,height,width)
spatial_image2=masking2(noise_image,height,width)
med_original=PSNR(image,spatial_image2)
plt.subplot(2,2,3)
plt.imshow(spatial_image,cmap='gray')
plt.subplot(2,2,4)
plt.imshow(spatial_image2,cmap='gray')
plt.title(f'Noisy image - PSNR: {med_original:.2f} dB')
plt.tight_layout()
plt.show()



