import matplotlib.pyplot as plt
import numpy as np
import cv2
st_element=np.ones((3,3))

image= cv2.imread('imgg/fingerprint.png', cv2.IMREAD_GRAYSCALE)
image=cv2.resize(image,(512,512))
height,width=image.shape


def erosion_op(image,st_element):

    erosion_img=np.copy(image)
    for i in range(1,height-1):
        for j in range(1,width-1):
            erosion_img[i,j]=np.min(image[i-1:i+2,j-1:j+2]*st_element)
    return erosion_img
def dilation_op(image,st_element):
    
    dialation_img=np.copy(image)
    for i in range(1,height-1):
        for j in range(1,width-1):
            dialation_img[i,j]=np.max(image[i-1:i+2,j-1:j+2]*st_element)
    return dialation_img


#  if np.all(image[i - 1:i + 2, j - 1:j + 2] >= st_element):
#                 erosion_img[i, j] = 255

boundary_img=image-erosion_op(image,st_element)
# closing_img=erosion_op(dilation_op(image,st_element),st_element)
plt.subplot(221)
plt.imshow(image,cmap='gray')
plt.title('original image')
plt.axis('off')
plt.subplot(223)
plt.imshow(boundary_img,cmap='gray')
plt.title('opening image')
plt.axis('off')
# plt.subplot(224)
# plt.imshow(closing_img,cmap='gray')
# plt.title('closing image')
# plt.axis('off')
plt.tight_layout()
plt.show()



#  okk 20-10-23