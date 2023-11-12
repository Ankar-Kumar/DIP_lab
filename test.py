a=[1,2,4,3]
print(a[1:6])
import numpy as np

# Create a 1D array
signal = np.array([1, 2, 3, 4, 5, 6,7,8])

# Compute the FFT
fft_result = np.fft.fft(signal)

# Shift the zero frequency component to the center
shifted_result = np.fft.fftshift(fft_result)

print("Original FFT Result:")
print(abs(fft_result))
print("\nShifted FFT Result:")
print(abs(shifted_result))
