import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('imgg/Characters Test Pattern 688x688.tif', 0)
image = cv2.resize(img, (512, 512))

height, width = image.shape

mean = 10
stddev = 25
noise = np.random.normal(mean ,stddev, image.shape).astype(np.uint8)

noisy_image = cv2.add(image, noise)

F = np.fft.fftshift(np.fft.fft2(noisy_image))

D0 = 5  # Cutoff frequency
n =15   # how many
dim= int(np.ceil(np.sqrt(n)))


  
(row, column) = F.shape
D=np.zeros((row,column))
for u in range(row):
        for v in range(column):            
            D[u,v]=np.sqrt( (u - row/2)**2 + (v - column/2)**2) # some problem occured if use only D


for i in range(n):
    idlf= D<=D0
    foutput_img = F* idlf
    # tmp_img = np.fft.ifft2(np.fft.ifftshift(foutput_img))
    # idlf_img = np.abs(tmp_img)

    tmp_img = np.abs(np.fft.ifft2(foutput_img))
    idlf_img = tmp_img #/255
    
    plt.subplot(dim,dim,i+1)
    plt.imshow(idlf_img,cmap='gray')
    plt.title(f'IDLF img when D0= {D0}')
    plt.axis('off')
    plt.tight_layout()
    D0 = D0 + 5


plt.show()
