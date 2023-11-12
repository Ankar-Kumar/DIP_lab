import matplotlib.pyplot as plt
import numpy as np
import cv2
st_element=np.ones((3,3))
st_element2=np.ones((5,5))
image= cv2.imread('imgg/boundary.png', cv2.IMREAD_GRAYSCALE)
image=cv2.resize(image,(512,512))
height,width=image.shape


def erosion_op(image,st_element):

    erosion_img=np.copy(image)
    for i in range(1,height-1):
        for j in range(1,width-1):
            erosion_img[i,j]=np.min(image[i-1:i+2,j-1:j+2]*st_element)
    return erosion_img

def erosion_op2(image,st_element):

    erosion_img=np.copy(image)
    for i in range(2, height-2):
        for j in range(2, width-2):
            erosion_img[i, j] = np.min(image[i-2:i+3, j-2:j+3] * st_element)

    return erosion_img
#  if np.all(image[i - 1:i + 2, j - 1:j + 2] >= st_element):
#                 erosion_img[i, j] = 255

boundary_img=image-erosion_op(image,st_element)
boundary_img2=image-erosion_op2(image,st_element2)
plt.subplot(221)
plt.imshow(image,cmap='gray')
plt.title('original image')
plt.axis('off')
plt.subplot(223)
plt.imshow(boundary_img,cmap='gray')
plt.title('3X3')
plt.axis('off')
plt.subplot(224)
plt.imshow(boundary_img2,cmap='gray')
plt.title('5X5')
plt.axis('off')
plt.tight_layout()
plt.show()



#  okk 20-10-23