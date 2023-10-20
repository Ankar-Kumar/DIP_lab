import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the grayscale image
original_image = cv2.imread('imgg/fingerprint.png')
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
resized_image = cv2.resize(gray_image, (512, 512))

# Plot the original image
plt.subplot(2, 2, 1)
plt.imshow(resized_image, cmap='gray')
plt.title('Original image')

# Define the structuring element (a 5x5 rectangular kernel)
se = np.ones((5, 5), np.uint8)

# Function to perform manual erosion
def manual_erode(image, se):
    rows, cols = image.shape
    eroded_image = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(2, rows - 2):
        for j in range(2, cols - 2):
            if np.all(image[i - 2:i + 3, j - 2:j + 3] >= se):
                eroded_image[i, j] = 255
    return eroded_image

# Perform manual erosion
eroded_image = manual_erode(resized_image, se)

# Calculate the boundary image
boundary_image = resized_image - eroded_image

# Plot the boundary image
plt.subplot(2, 2, 2)
plt.imshow(boundary_image, cmap='gray')
plt.title('Boundary of the image')

plt.tight_layout()
plt.show()
