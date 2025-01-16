# Texture Analysis
from skimage.feature import greycomatrix, greycoprops
from skimage import color
import numpy as np

### Convert the wound area to grayscale
gray_wound_image = cv2.cvtColor(wound_image, cv2.COLOR_BGR2GRAY)

### Compute GLCM
glcm = greycomatrix(gray_wound_image, distances=[1], angles=[0], symmetric=True, normed=True)

### Extract GLCM features
contrast = greycoprops(glcm, 'contrast')[0, 0]
homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
print(f"Contrast: {contrast}, Homogeneity: {homogeneity}")

# Color Analysis
### Convert image to HSV color space
hsv_image = cv2.cvtColor(wound_image, cv2.COLOR_BGR2HSV)

### Define color range for yellow (e.g., slough)
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])
mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

### Calculate percentage of yellow in wound area
yellow_pixels = np.count_nonzero(mask_yellow)
total_pixels = mask_yellow.size
yellow_percentage = (yellow_pixels / total_pixels) * 100
print(f"Yellow area percentage: {yellow_percentage}%")

# Surface Area (Wound Size)

import cv2
import cv2.aruco as aruco

### Load image
image = cv2.imread('foot_image.jpg')

### Detect ArUco markers
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters_create()
corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

### If markers are found
if len(corners) > 0:
    aruco.drawDetectedMarkers(image, corners, ids)

    # Calculate the distance between the corners of the marker
    marker_width_in_pixels = cv2.norm(corners[0][0][0] - corners[0][0][1], cv2.NORM_L2)
    real_marker_width = 5  # cm (known size of the ArUco marker)

    # Calculate pixel-to-centimeter ratio
    pixels_per_cm = marker_width_in_pixels / real_marker_width
    print(f"Pixels per cm: {pixels_per_cm}")



