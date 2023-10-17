import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
image= cv2.imread('imgg/images.jpg', cv2.IMREAD_GRAYSCALE)
final_img=cv2.resize(image,(512,512))

height,width=final_img.shape

msb_mask =224 
gray_image_msb =final_img & msb_mask

# diff=cv2.absdiff(final_img,gray_image_msb)
diff=np.zeros_like(final_img)
# diff = final_img.copy()
for i in range(height):
    for j in range(width):
        pixel=abs(int(final_img[i,j]-gray_image_msb[i,j]))
        diff[i,j]=pixel

cv2.imshow('difference image', diff)
cv2.imshow('original image',final_img)
cv2.imshow('msb image',gray_image_msb)
cv2.waitKey()


#   this code is checked and result alright