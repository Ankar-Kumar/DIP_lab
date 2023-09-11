import numpy as np
import cv2  # OpenCV is used for visualization purposes

import matplotlib.pyplot as plt
# Load the image
image = cv2.imread('test.jpg',cv2.IMREAD_GRAYSCALE)
initail_img=np.array(image)
height,width=image.shape
height=height//2
width=width//2
new_image=np.zeros((height,width),dtype=np.uint8)
for i in range(height):
    c=0
    for j in range(width):        
        rstart=i*2
        rend=rstart+2
        cstart=j*2
        cend=cstart+2
        pixel=np.mean(initail_img[rstart:rend,cstart:cend],axis=(0,1))
        new_image[i,j]=pixel.astype(np.uint8)



cv2.imshow('original',image)
cv2.imshow('scaled',new_image)
cv2.waitKey(0)

        
