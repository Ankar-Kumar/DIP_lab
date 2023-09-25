import numpy as np

def pad_image(image, pad_height, pad_width):
    
    # Get the dimensions of the input image
    height, width = image.shape

    # Calculate the new dimensions after padding
    new_height = height + 2 * pad_height
    new_width = width + 2 * pad_width

    # Create a new array filled with zeros (padded_image)
    padded_image = np.zeros((new_height, new_width), dtype=image.dtype)

    # Copy the input image into the center of the padded_image
    padded_image[pad_height:pad_height + height, pad_width:pad_width + width] = image

    return padded_image

# Example usage:
# pad_height and pad_width specify the amount of padding on each side
image = np.array([[1, 2], [3, 4]])
pad_height = 2
pad_width = 2
padded_image = pad_image(image, pad_height, pad_width)
print(padded_image)
