import cv2
import numpy as np

# Load the image
img = cv2.imread('input_image.jpg')

# Calculate the center of rotation (x, y)
center_x, center_y = img.shape[1] // 2, img.shape[0] // 2

# Create a rotation matrix
#angle = np.deg2rad(45)  # 45 degrees in radians
angle = 45
rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)

# Apply the rotation matrix to the image
rotated_img = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))

# Display the original and rotated images
cv2.imshow('Original', img)
cv2.imshow('Rotated', rotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()