import matplotlib.pyplot as plt
import numpy as np
import cv2
st_elemen=np.ones((3,3))

image= cv2.imread('imgg/fingerprint.png', cv2.IMREAD_GRAYSCALE)
image=cv2.resize(image,(512,512))
height,width=image.shape

erosion_img=np.copy(image)
dialation_img=np.copy(image)


for i in range(1,height-1):
    for j in range(1,width-1):
        erosion_img[i,j]=np.min(image[i-1:i+2,j-1:j+2]*st_elemen)


for i in range(1,height-1):
    for j in range(1,width-1):
        dialation_img[i,j]=np.max(image[i-1:i+2,j-1:j+2]*st_elemen)


#  if np.all(image[i - 1:i + 2, j - 1:j + 2] >= st_elemen):
#                 erosion_img[i, j] = 255

plt.subplot(221)
plt.imshow(image,cmap='gray')
plt.title('original image')
plt.axis('off')
plt.subplot(223)
plt.imshow(erosion_img,cmap='gray')
plt.title('erosion image')
plt.axis('off')
plt.subplot(224)
plt.imshow(dialation_img,cmap='gray')
plt.title('dialation image')
plt.axis('off')
plt.tight_layout()
plt.show()



#  okk 17-10-23