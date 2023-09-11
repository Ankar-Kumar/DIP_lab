import numpy as np
import cv2  # OpenCV is used for visualization purposes

# Load the image
image = cv2.imread('unty.jpg', cv2.IMREAD_GRAYSCALE)

# Define upsampling factor
upscale_factor = 2

# Upsampling
def upscale_image(image, scale_factor):
    height, width = image.shape
    new_height = height * scale_factor
    new_width = width * scale_factor
    new_image = np.zeros((new_height, new_width), dtype=np.uint8)
# 0   0
# 1   0
# 2   1
# 3   1
# 4   2
# 5   2
    for i in range(new_height):
        for j in range(new_width):
            src_i = int(i / scale_factor)
            src_j = int(j / scale_factor)
            print(j,' ',src_j)
            if j==5:
                break
            new_image[i, j] = image[src_i, src_j]
        break
    
    return new_image

upscaled_image = upscale_image(image, upscale_factor)

# Define downsampling factor
downscale_factor = 2

# Downsampling
def downscale_image(image, scale_factor):
    height, width = image.shape
    new_height = int(height / scale_factor)
    new_width = int(width / scale_factor)
    new_image = np.zeros((new_height, new_width), dtype=np.uint8)
    
    for i in range(new_height):
        for j in range(new_width):
            src_i = i * scale_factor # scale factor por por value nicci
            src_j = j * scale_factor
            new_image[i, j] = image[src_i, src_j]
    
    return new_image

downscaled_image = downscale_image(image, downscale_factor)

# Display the images using OpenCV
cv2.imshow('Original Image', image)
cv2.imshow('Upscaled Image', upscaled_image)
cv2.imshow('Downscaled Image', downscaled_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
