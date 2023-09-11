# task1 split image into 4/8 parts 21-08

import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
im=Image.open('unty.jpg')

ndivheight=2
ndivwidth=4
totallDivision=ndivwidth*ndivheight
# print(im.size)
arr=np.array(im)
height,width,_ =arr.shape
h,w=height,width
height=height//ndivheight
width=width//ndivwidth
split_images=np.empty((totallDivision,height,width,3),dtype=np.uint8)

for i in range(ndivheight):
    for j in range(ndivwidth):
        split_images[i*ndivwidth+j]=arr[i*height:(i+1)*height,j*width:(j+1)*width]

# another method
# split_images=[]
# splitt=arr[i*height:(i+1)*height,j*width:(j+1)*width]
#         split_images.append(splitt)


plt.figure(1)
c=1
for i,split in enumerate(split_images):
    plt.subplot(ndivheight,ndivwidth,i+1)
    plt.imshow(split)
    if c==1:
        plt.title("splitting images into parts")
    c+=1
    plt.axis('off')


#  show the actual image from the splitting images

actual_image=np.zeros_like(im)
for i,split in enumerate(split_images):
    heights=i//ndivwidth
    widths=i%ndivwidth
    actual_image[heights*height:(heights+1)*height,widths*width:(widths+1)*width]=split

plt.figure(2)
plt.imshow(actual_image)
plt.axis('off')
plt.title("actual image back")
plt.show()

