# README

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

- Start off by using about 20 chessboard images calibrate my camera
- Obtain `image points` and `object points` using **cv2.findChessboardCorners**
<img align="center" src="https://github.com/x65han/Advanced-Lane-Detection/blob/master/assets/report/chessboard.png?raw=true" width="100%" />

- Using **cv2.undistort**, I obtained the following result on `chessboard` images
<img align="center" src="https://github.com/x65han/Advanced-Lane-Detection/blob/master/assets/report/undistorted_chessboard.png?raw=true" width="100%" />

#### 1. Provide an example of a distortion-corrected image.

- Using **cv2.undistort**, I obtained the following result on `highway` images
- Distortion is most obvious on the car hood and the trees on the left
<img align="center" src="https://github.com/x65han/Advanced-Lane-Detection/blob/master/assets/report/undistorted.jpg?raw=true" />

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

- I explored with a variety of colorspaces
    - **R** **G** **B**
    - **H** **S** **V**
    - **L** **A** **B**
    
<img align="center" src="https://github.com/x65han/Advanced-Lane-Detection/blob/master/assets/report/colorspaces.jpg?raw=true" />

- Then I created this **pre-processing pipeline**
    - `undistort` image
    - `unwarp` image
    - apply `L-Channel threshold`
    - apply `B-Channel threshold`
    - `Combine` both channels to generate `binary image`

<img align="center" src="https://github.com/x65han/Advanced-Lane-Detection/blob/master/assets/report/pipeline.jpg?raw=true" />

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

With perspective transform, I used the following **source points** & **destination points**
```python
src = np.float32([(575,464), (707,464), 
                  (258,682),(1049,682)])

dst = np.float32([(450,0), (width-450,0),
                  (450,height), (width-450,height)])
```

The code for my `unwarp` function
```python
def unwarp(img, src, dist):
    height, width = img.shape[:2]
    # Calculate transformation matrix
    matrix = cv2.getPerspectiveTransform(src, dist)
    inverse_matrix = cv2.getPerspectiveTransform(dist, src)
    # Apply transformation
    warped = cv2.warpPerspective(img, matrix, (width, height), flags=cv2.INTER_LINEAR)
    return warped, matrix, inverse_matrix
```
Here is an example of utilizing `unwarp` function:
<img align="center" src="https://github.com/x65han/Advanced-Lane-Detection/blob/master/assets/report/unwarped.jpg?raw=true" />


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

- To better detect the lanes, I used `histogram` technique to detect vertical pixels
<img align="center" src="https://github.com/x65han/Advanced-Lane-Detection/blob/master/assets/report/sliding_window_histogram.jpg?raw=true" />

- In `helper.py`, I defined a `sliding_window_polyfit` identify lane-line pixels
    - only look at left quarter and right quarter to find pixels using `histogram`
    - using sliding window to identify lanes

<img align="center" src="https://github.com/x65han/Advanced-Lane-Detection/blob/master/assets/report/sliding_window.jpg?raw=true" />
    
- In `helper.py`, I also defined a `polyfit_using_previous_fit` to polyfit current line based on previous result
    - Consecutive lane detection should be approximately close to each other (chronologically speaking)
    - So if the newly detected lane is within 100px from last detection
    - use this function to detect lanes based on previous result

<img align="center" src="https://github.com/x65han/Advanced-Lane-Detection/blob/master/assets/report/fit_lane_highlight.jpg?raw=true" />

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

- In `helper.py`, I defined a `calc_curvature_and_center_dist` function to return
    - left curvature
    - right curvature
    - distance from center
- Here are the steps
    - Define conversion in x and y from pixels to meters
    - Define y-value where we want radius of curvature
    - Identify x and y positions of all non-zero pixels
    - extract left and right line pixel positions
    - use `np.polyfit` to fit new polynomials to x,y in meters
    - Compute distance from center in image x midpoint -> mean of left_fit and right_fit intercepts


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

- In `helper.py`, I defined a `draw_lane` function to
<img align="center" src="https://github.com/x65han/Advanced-Lane-Detection/blob/master/assets/report/draw_lane.jpg?raw=true" />

- In `helper.py`, I defined a `draw_data` function to
<img align="center" src="https://github.com/x65han/Advanced-Lane-Detection/blob/master/assets/report/draw_data.jpg?raw=true" />

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://github.com/x65han/Advanced-Lane-Detection/blob/master/outputs/project_video_output.mp4?raw=true)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

When there is shadows, my pipeline will likely to fail. Shadows creates extra difficult in term of finding the lanes. My pipeline will be more robust if there a way of filtering out the shadows or distinguish between shadows and lanes.
