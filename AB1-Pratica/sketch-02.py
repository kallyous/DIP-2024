import cv2
import numpy as np

# Create a blank image with a white background
img = np.zeros((512, 512, 3), dtype=np.uint8)

# Draw a circle with center (256, 256), radius 100, and blue color
cv2.circle(img, (256, 256), 100, (255, 0, 0), 2)

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply thresholding to convert to binary (black and white)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Display the resulting binary circle
cv2.imshow('Binary Circle', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()