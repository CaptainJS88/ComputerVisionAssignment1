import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

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

# print(myArray)
# print(cv2.__version__)

# print(type(img))   # Should be <class 'numpy.ndarray'>
# print(img.shape)  # Should be (Height, Width, Channels)
# print(img.dtype)  # Should be uint8
# print(img)
# print(img_rgb)
