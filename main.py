import numpy as np
import cv2
import glob
import pickle
from helper import displayImages, pipeline, undistort, unwarp, sliding_window_polyfit
import matplotlib.pyplot as plt

# Regex: find all calibration files
input_images = sorted(glob.glob("./assets/camera_cal/calibration*.jpg"))
print("There are {} images to process for calibration".format(len(input_images)))
output_images = []
output_titles = []
# used later to calibrate cameras
obj_points = []
img_points = []
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
obj_point = np.zeros((6*9, 3), np.float32)
obj_point[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

for file_name in input_images:
    # Load images
    image = cv2.imread(file_name)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray_image, (9, 6), None)
    # If found, draw & display
    if ret == True:
        img_points.append(corners)
        obj_points.append(obj_point)
        
        output_image = cv2.drawChessboardCorners(image, (9, 6), corners, ret)
        output_images.append(output_image)
        output_titles.append(file_name)

displayImages(output_images, output_titles, file_name="assets/report/chessboard.png")
        
# Undisort selected calibration images
image_path = sorted(glob.glob("./assets/camera_cal/calibration*.jpg"))[0]
image = cv2.imread(image_path)
image_size = (image.shape[1], image.shape[0])

# Calibrate Camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, image_size, None, None)
# Save camera calibration data for later use (Ignore rvecs & tvecs)
output_pickle = {}
output_pickle["mtx"] = mtx
output_pickle["dist"] = dist
pickle.dump(output_pickle, open("calibration.p", "wb"))

# Display Undistorted Chessboard images
destination_image = undistort(image, mtx, dist)
displayImages([image, destination_image], ["Original Image", "Un-distorted Image"], file_name="assets/report/undistorted_chessboard.png")

# Pick a image to demonstrate pipline
image = cv2.imread('./assets/inputs/bridge.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Undistort Images
undistorted = undistort(image, mtx, dist)
displayImages([image, undistorted], ["Original Image", "Undistorted Image"], file_name="./assets/report/undistorted.jpg")

height, width = undistorted.shape[:2]

# define the source and destination points of transformation
src = np.float32([(575,464), (707,464),  (258,682),(1049,682)])
dst = np.float32([(450,0), (width-450,0), (450,height), (width-450,height)])
mask_src = np.array([[(320,682), (640,464), (720,464), (1049,682)]], dtype=np.int32)

# unwarp
unwarped, matrix, inverse_matrix = unwarp(undistorted, src, dst)

# Draw & Display
masked = cv2.polylines(undistorted, mask_src, isClosed=True, color=(0, 255, 0))
displayImages([masked, unwarped], ["Undistorted Image", "Unwarped Image"], file_name="./assets/report/unwarped.jpg")

output_images = []
output_titles = []

# RGB
output_images.append(unwarped[:,:,0])
output_titles.append("R")
output_images.append(unwarped[:,:,1])
output_titles.append("G")
output_images.append(unwarped[:,:,2])
output_titles.append("B")
# HSV
hsv = cv2.cvtColor(unwarped, cv2.COLOR_RGB2HSV)
output_images.append(hsv[:,:,0])
output_titles.append("H")
output_images.append(hsv[:,:,1])
output_titles.append("S")
output_images.append(hsv[:,:,2])
output_titles.append("V")
# HLS
hls = cv2.cvtColor(unwarped, cv2.COLOR_RGB2HLS)
output_images.append(hls[:,:,0])
output_titles.append("H")
output_images.append(hls[:,:,1])
output_titles.append("L")
output_images.append(hls[:,:,2])
output_titles.append("S")
# LAB
lab = cv2.cvtColor(unwarped, cv2.COLOR_RGB2Lab)
output_images.append(lab[:,:,0])
output_titles.append("L")
output_images.append(lab[:,:,1])
output_titles.append("A")
output_images.append(lab[:,:,2])
output_titles.append("B")

displayImages(output_images, output_titles, file_name="./assets/report/colorspaces.jpg", gray=True, columns=3)

# Pipeline
images = glob.glob("./assets/inputs/*.jpg")

output_images = []
output_titles = []
output_inverse_matrix = []

for file_name in images:
    image = cv2.imread(file_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output_image, inverse_matrix = pipeline(image, mtx, dist, src, dst)
    output_inverse_matrix.append(inverse_matrix)
    output_images.append(image)
    output_titles.append(file_name)
    output_images.append(output_image)
    output_titles.append("After pipeline")
    
displayImages(output_images, output_titles, file_name="./assets/report/pipeline.jpg")

# Sliding Window Polyfit
# visualize the result on example image
exampleImg = cv2.imread('./assets/inputs/test2.jpg')
exampleImg = cv2.cvtColor(exampleImg, cv2.COLOR_BGR2RGB)
exampleImg_bin, Minv = pipeline(exampleImg, mtx, dist, src, dst)
    
left_fit, right_fit, left_lane_inds, right_lane_inds, visualization_data = sliding_window_polyfit(exampleImg_bin)

h = exampleImg.shape[0]
left_fit_x_int = left_fit[0]*h**2 + left_fit[1]*h + left_fit[2]
right_fit_x_int = right_fit[0]*h**2 + right_fit[1]*h + right_fit[2]
#print('fit x-intercepts:', left_fit_x_int, right_fit_x_int)

rectangles = visualization_data[0]
histogram = visualization_data[1]

# Create an output image to draw on and  visualize the result
out_img = np.uint8(np.dstack((exampleImg_bin, exampleImg_bin, exampleImg_bin))*255)
# Generate x and y values for plotting
ploty = np.linspace(0, exampleImg_bin.shape[0]-1, exampleImg_bin.shape[0] )
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
for rect in rectangles:
# Draw the windows on the visualization image
    cv2.rectangle(out_img,(rect[2],rect[0]),(rect[3],rect[1]),(0,255,0), 2) 
    cv2.rectangle(out_img,(rect[4],rect[0]),(rect[5],rect[1]),(0,255,0), 2) 
# Identify the x and y positions of all nonzero pixels in the image
nonzero = exampleImg_bin.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])
out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [100, 200, 255]
plt.imshow(out_img)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)

print('---')
