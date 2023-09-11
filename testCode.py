import numpy as np
import cv2  # OpenCV is used for visualization purposes

# Load the image
image = cv2.imread('lab.jpg',cv2.IMREAD_GRAYSCALE)
arr=np.array(image)
# print(image.shape)
print(arr[:2,:50])