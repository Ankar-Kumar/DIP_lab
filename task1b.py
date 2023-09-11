import numpy as np
import cv2  # OpenCV is used for visualization purposes
import matplotlib.pyplot as plt
import math
# Load the image
image = cv2.imread('imgg/cameraman.jpg',cv2.IMREAD_GRAYSCALE)
img=cv2.resize(image,(512,512))
height,width=img.shape
level=8
dim=3
plt.subplot(dim,dim,1)
plt.imshow(img,cmap='gray')
plt.title('8')

# tmp_img= np.zeros((height, width), dtype=np.double)

for i in range(1,level):
    tmp_img = np.zeros((height, width))
    for j in range(height):
        for k in range(width):
            tmp =img[j, k] // 2**i #  bitwise right shift operation to remove the least significant i bits
            tmp_img[j, k] = tmp 
    # tmp_img=cv2.resize(tmp_img,(512,512))
    plt.subplot(dim,dim,i+1)
    plt.imshow(tmp_img,cmap='gray')
    plt.title(f'{level-i} bit')
    plt. subplots_adjust(wspace=0.2)

plt.tight_layout()
plt.show()

# last checked in 3.32 pm is alright