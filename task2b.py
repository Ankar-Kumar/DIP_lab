import numpy as np
import cv2
import matplotlib.pyplot as plt
image= cv2.imread('imgg/tain.png', cv2.IMREAD_GRAYSCALE)
final_img=cv2.resize(image,(512,512))
import math
height,width=final_img.shape

gama=0.6

##  option 1 for inverse log 
c=255/(np.log(1+255))
# log_image=c*np.log(final_img+1)
# log_image=np.array(log_image,dtype=np.uint8)


# power=np.power(final_img/255.0,gama)*c
# power_img=np.uint8(power)
tmp_img=np.zeros((height,width))
power_img=np.zeros((height,width))
inv_log=np.zeros((height,width))

for i in range(height):
    for j in range(width):
        power_img[i,j]=(final_img[i,j]**gama)
        # tmp_img[i,j]=c*np.log(1.0+final_img[i,j])
        inv_log[i,j]=np.exp(final_img[i,j] / c) -1
# inv_log=np.exp(final_img / c) -1

plt.subplot(1,3,1)
plt.imshow(final_img,cmap='gray')
plt.title('original image')
plt.subplot(1,3,2)
plt.imshow(inv_log,cmap='gray') 
plt.title('inverse log image')
plt.subplot(1,3,3) 
plt.imshow(power_img,cmap='gray')
plt.title('power image ')
plt.show()