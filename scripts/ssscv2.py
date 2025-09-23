# Python program to identify
#color in images

# Importing the libraries OpenCV and numpy
import cv2
import numpy as np

# Read the images
img = cv2.imread("images/DP167067.jpg")

# Resizing the image
image = cv2.resize(img, (700, 600))

# Convert Image to Image HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Print HSV values
print("HSV shape:", hsv.shape)
print("HSV min values:", hsv.min(axis=(0,1)))
print("HSV max values:", hsv.max(axis=(0,1)))
print("HSV mean values:", hsv.mean(axis=(0,1)))
print("\nFirst few HSV pixel values:")
print(hsv[:5, :5])  # Print first 5x5 pixels

# Defining lower and upper bound HSV values
lower = np.array([50, 100, 100])
upper = np.array([70, 255, 255])

# Defining mask for detecting color
mask = cv2.inRange(hsv, lower, upper)

# Display Image and Mask
# cv2.imshow("Image", image)
# cv2.imshow("Mask", mask)

# Make python sleep for unlimited time
cv2.waitKey(0)