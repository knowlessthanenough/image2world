import numpy as np
import cv2

# Termination criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (assuming checkerboard squares are 25x25mm)
square_size = 130  # Replace with your checkerboard square size
objp = np.zeros((5*4,3), np.float32)
objp[:,:2] = np.mgrid[0:5,0:4].T.reshape(-1,2) * square_size

# Load camera matrix and distortion coefficients
mtx = np.load('data/rgb_intrinsic_matrix.npy')
# set a zero distortion coefficient: 0,0,0,0,0
dist = np.zeros((5,1), np.float32)


# Read the image containing the checkerboard
img = cv2.imread('data/test_color.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find the chess board corners
ret, corners = cv2.findChessboardCorners(gray, (5,4), None)

if ret == True:
    # Refine corner positions
    corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

    # Find the rotation and translation vectors
    _, rvecs, tvecs, _ = cv2.solvePnPRansac(objp, corners2, mtx, dist)

    # Save the external parameters
    np.save('data/rgb_rotation_vector.npy', rvecs)
    print(rvecs)
    np.save('data/rgb_translation_vector.npy', tvecs)
    print(tvecs)
    # Display the image with detected corners
    cv2.drawChessboardCorners(img, (5,4), corners2, ret)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Checkerboard not found in the image")

# The rotation vector is saved as rotation_vector.npy and translation vector as translation_vector.npy