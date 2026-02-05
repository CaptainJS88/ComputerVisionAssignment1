import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import copy

# Testing if np and cv are successfully imported. 
myArray = np.array([1,2,3,4])

# Importing colors.png using matplotlib
img = cv2.imread('./Images/colors.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Extracting channels
R = img_rgb[:, :, 0]
G = img_rgb[:, :, 1]
B = img_rgb[:, :, 2]

# Creating the figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Display Red
axes[0].imshow(R, cmap='gray')
axes[0].set_title('Red Channel')
axes[0].axis('off') # Optional: hides the pixel coordinates

# Display Green
axes[1].imshow(G, cmap='gray')
axes[1].set_title('Green Channel')
axes[1].axis('off')

# Display Blue
axes[2].imshow(B, cmap='gray')
axes[2].set_title('Blue Channel')
axes[2].axis('off')

plt.show()

# Binary masking 
mask = (R >= 100) & (R <= 255) & (B >= 0) & (B <= 50) & (G >= 0) & (G <= 50)

masked_image = img_rgb.copy()
masked_image[~mask] = 0 # Black

plt.imshow(masked_image)
plt.title("Masked Image (Red Only)")
plt.show()

# Grayscale image

# 1. Convert channels to float to allow values higher than 255
R_f = R.astype(float)
G_f = G.astype(float)
B_f = B.astype(float)

# 2. Apply the formula to the whole array at once
grayscale_f = (R_f + G_f + B_f) / 3

# 3. Convert back to uint8 so matplotlib can display it as an image
grayscale = grayscale_f.astype('uint8')

# 4. Display the result
plt.imshow(grayscale, cmap='gray')
plt.title("Grayscale via Average Formula")
plt.axis('off')
plt.show()

# Importing apple.jpg and stop-sign.jpg and creating their copies. 
apple_img = cv2.imread('./Images/apple.jpg')
apple_img_rgb = cv2.cvtColor(apple_img, cv2.COLOR_BGR2RGB)

stop_sign_img = cv2.imread('./Images/stop-sign.jpg')
stop_sign_img_rgb =cv2.cvtColor(stop_sign_img, cv2.COLOR_BGR2RGB)

#Creating deep copies of the apple and stop sign image
apple_img_copy = copy.deepcopy(apple_img_rgb)
stop_sign_img_copy = copy.deepcopy(stop_sign_img_rgb)

# Manual function to convert RGB to HSV

def manual_rgb_to_hsv(rgb_img):
    # Normalizing to 0-1 range
    img = rgb_img.astype('float32') / 255.0

    # Separating the channels
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    # Calculating V (Max) and Delta (Max - Min)
    cmax = np.max(img, axis=2)
    cmin = np.min(img, axis=2)
    delta = cmax - cmin

    # Initialize H, S, V arrays with zeros
    H = np.zeros_like(cmax)
    S = np.zeros_like(cmax)
    V = cmax # Value is just the Max component

    # Calculate Saturation: 0 if cmax is 0, otherwise delta / cmax
    # Use np.where to handle division by zero safely
    S = np.where(cmax == 0, 0, delta / cmax)

    # Calculate Hue
    # Where Delta is 0, Hue is 0 (undefined/grayscale)
    # Masking is needed to calculate Hue only where Delta != 0
    indices = delta > 0
    
    # R is max
    idx = (R == cmax) & indices
    H[idx] = (G[idx] - B[idx]) / delta[idx] % 6
    
    # G is max
    idx = (G == cmax) & indices
    H[idx] = (B[idx] - R[idx]) / delta[idx] + 2
    
    # B is max
    idx = (B == cmax) & indices
    H[idx] = (R[idx] - G[idx]) / delta[idx] + 4
    
    H = H * 60 # Convert to degrees
    
    # OpenCV uses 0-179 for Hue (to fit in uint8), 0-255 for S and V
    # So we scale our 0-360 Hue to 0-179
    H = H / 2 
    S = S * 255
    V = V * 255
    
    # Merge back into a 3D image
    hsv_image = np.dstack((H, S, V)).astype('uint8')
    return hsv_image

# Converting the copies to HSV color space
apple_img_copy_hsv = manual_rgb_to_hsv(apple_img_copy)
stop_sign_img_copy_hsv = manual_rgb_to_hsv(stop_sign_img_copy)

print(apple_img_copy_hsv, "apple hsv")
print(stop_sign_img_copy_hsv,"stop sign hsv")

# Use binary operations to extract the pixels containing apple and the stop sign respectively
H_APPLE = apple_img_copy_hsv[:, :, 0]
S_APPLE = apple_img_copy_hsv[:, :, 1]
V_APPLE = apple_img_copy_hsv[:, :, 2]

H_STOP_SIGN = stop_sign_img_copy_hsv[:, :, 0]
S_STOP_SIGN = stop_sign_img_copy_hsv[:, :, 1]
V_STOP_SIGN = stop_sign_img_copy_hsv[:, :, 2]

# Creating mask for the green apple
green_apple_mask = (H_APPLE >= 30) & (H_APPLE <= 60) & (S_APPLE >= 30) & (V_APPLE >= 20)
apple_img_copy_hsv[~green_apple_mask] = 0

# Creating mask for the stop sign, here, its best to simply subtract the blue sky I feel
stop_sign_mask = (H_STOP_SIGN >= 90) & (H_STOP_SIGN <= 130) & (S_STOP_SIGN >= 50)
stop_sign_img_copy_hsv[stop_sign_mask] = 0

# Converting apple and stop sign image back to rgb
final_apple_img = cv2.cvtColor(apple_img_copy_hsv, cv2.COLOR_HSV2RGB)
final_stop_sign_img = cv2.cvtColor(stop_sign_img_copy_hsv, cv2.COLOR_HSV2RGB)


# Show Apple
plt.imshow(final_apple_img)
plt.title("Extracted Green Apple")
plt.axis('off')
plt.show()

# Show Stop sign
plt.imshow(final_stop_sign_img)
plt.title("Extracted Stop sign image")
plt.axis('off')
plt.show()

# Show together, with orignal, masked hsv and masked rgb
fig, axes = plt.subplots(2, 3)

# Row 1: Apple
axes[0, 0].imshow(apple_img_rgb)
axes[0, 0].set_title("Original Apple")
axes[0, 0].axis('off')

axes[0, 1].imshow(apple_img_copy_hsv)
axes[0, 1].set_title("Masked Apple (HSV)")
axes[0, 1].axis('off')

axes[0, 2].imshow(final_apple_img)
axes[0, 2].set_title("Masked Apple (RGB)")
axes[0, 2].axis('off')

# Row 2: Stop Sign
axes[1, 0].imshow(stop_sign_img_rgb)
axes[1, 0].set_title("Original Stop Sign")
axes[1, 0].axis('off')

axes[1, 1].imshow(stop_sign_img_copy_hsv)
axes[1, 1].set_title("Masked Sign (HSV)")
axes[1, 1].axis('off')

axes[1, 2].imshow(final_stop_sign_img)
axes[1, 2].set_title("Masked Sign (RGB)")
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()


# Question 3 - read bandnoise image, apply fourier transform and denoise

# Reading the image and converting to Grayscale
img_bandnoise = cv2.imread('./Images/bandnoise.png')
img_bandnoise_gray = cv2.cvtColor(img_bandnoise, cv2.COLOR_BGR2GRAY)

# Applying fourier transform to the array, and shifting 
img_bandnoise_fft = np.fft.fft2(img_bandnoise_gray)
img_bandnoise_fft_shift = np.fft.fftshift(img_bandnoise_fft)
magnitude_spectrum = 20 * np.log(np.abs(img_bandnoise_fft_shift))
img_bandnoise_fft_shift_copy = img_bandnoise_fft_shift.copy()

# Filtering the bright spots out of the fourier image
bright_spot_values = [50, 90, 130, 220, 260, 300]
for i in bright_spot_values:
    img_bandnoise_fft_shift_copy[:, (i-10):(i+10)] = 0

# Moving the filtered image back to corners from the center
f_ishift = np.fft.ifftshift(img_bandnoise_fft_shift_copy)

# Inverse FFT (Frequency -> Spatial)
img_bandnoise_back = np.fft.ifft2(f_ishift)

# Magnitude (Complex -> Real pixel values)
img_bandnoise_back = np.abs(img_bandnoise_back)

# Showing the plotted images
fig, axes = plt.subplots(1, 3)

# Plot 1: Original gray image
# We use axes[0] because 'axes' is a 1D list here
axes[0].set_title("Bandnoise Gray Image")
axes[0].imshow(img_bandnoise_gray, cmap='gray')

# Plot 2: Fourier image
# We use axes[1] for the second plot in the list
axes[1].set_title("Fourier Image")
axes[1].imshow(magnitude_spectrum, cmap='gray')

# Plot 3: Cleaned image
axes[2].set_title("Clean Image (Filtered)")
axes[2].imshow(img_bandnoise_back, cmap='gray')

plt.tight_layout()
plt.show()

# Question 4 - Dog image changes

# Importing the dog image

img_dog = cv2.imread('./Images/dog.jpg')
img_dog_rgb = cv2.cvtColor(img_dog, cv2.COLOR_BGR2RGB)
img_dog_float = img_dog_rgb.astype(np.float32)

# Applying the filters
img_dog_darker = np.clip(img_dog_float - 128, 0, 255).astype(np.uint8)
img_dog_low_contrast = (img_dog_float / 2).astype(np.uint8)
img_dog_inverted = 255 - img_dog
img_dog_brighter = np.clip(img_dog_float + 128, 0, 255).astype(np.uint8)
img_dog_high_contrast = np.clip(img_dog_float * 2, 0, 255).astype(np.uint8)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
# Flatten the axes array to make indexing easier (0 to 5)
ax = axes.flatten()

# 0. Original
ax[0].imshow(img_dog_rgb)
ax[0].set_title("Original")
ax[0].axis('off')

# 1. Darker (a)
ax[1].imshow(img_dog_darker)
ax[1].set_title("Darker (-128)")
ax[1].axis('off')

# 2. Low Contrast (b)
ax[2].imshow(img_dog_low_contrast)
ax[2].set_title("Low Contrast (/2)")
ax[2].axis('off')

# 3. Invert (c)
ax[3].imshow(img_dog_inverted)
ax[3].set_title("Inverted")
ax[3].axis('off')

# 4. Brighter (d)
ax[4].imshow(img_dog_brighter)
ax[4].set_title("Brighter (+128)")
ax[4].axis('off')

# 5. High Contrast (e)
ax[5].imshow(img_dog_high_contrast)
ax[5].set_title("High Contrast (*2)")
ax[5].axis('off')

plt.tight_layout()
plt.show()



