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



plt.imshow(final_apple_img)
plt.title("Extracted Green Apple")
plt.axis('off')
plt.show()

plt.imshow(final_stop_sign_img)
plt.title("Extracted Stop sign image")
plt.axis('off')
plt.show()









# print(myArray)
# print(cv2.__version__)

# print(type(img))   # Should be <class 'numpy.ndarray'>
# print(img.shape)  # Should be (Height, Width, Channels)
# print(img.dtype)  # Should be uint8
# print(img)
# print(img_rgb)
