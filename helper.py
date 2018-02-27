# Import Statements
import numpy as np
import cv2, glob, math, pickle
import matplotlib.pyplot as plt

# Define a class to handle all help functions
class Helper():
    def __init__(self):
        image = cv2.imread(glob.glob("./assets/inputs/*jpg")[0])
        # Global Variable: image size in px as pair (width, height)
        self.image_size = image.shape[:2]
        # Global Variable: image height in px
        self.image_height = self.image_size[1]
        # Global Variable: image width in px
        self.image_width = image.shape[0]
        # Global Variable: camera transformation matrix
        self.mtx = None
        # Global Variable: inverse camera transformation matrix
        self.inverse_matrix = None
        # Global Variable: camera transformation matrix
        self.dist = None
        ## Global Variable: source transformation matrix
        self.src = np.float32([(575, 464), (707, 464), (258, 682),(1049, 682)])
        ## Global Variable: destination transformation matrix
        self.dst = np.float32([(450, 0), (self.image_height - 450, 0), (450, self.image_height), (self.image_width - 450, self.image_height)])

    def display_and_save(self, images, titles, columns=2, file_name=False, gray=False):
        number_of_images = len(images)
        rows = math.ceil(number_of_images / columns)
        fig, axs = plt.subplots(rows, columns, figsize=(40, 10 * rows))
        fig.subplots_adjust(hspace=.2, wspace=.001)
        axs = axs.ravel()
        
        for i in range(columns * rows):
            axs[i].axis('off')

        for index, image in enumerate(images):
            if gray == True:
                axs[index].imshow(image, cmap='gray')
            else:
                axs[index].imshow(image)
            axs[index].set_title(titles[index], fontsize=30)
        
        if file_name:
            fig.savefig(file_name)
        # plt.show()

    def undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

    def unwarp(self, img, src, dist):
        height, width = img.shape[:2]
        # Calculate transformation matrix
        matrix = cv2.getPerspectiveTransform(src, dist)
        inverse_matrix = cv2.getPerspectiveTransform(dist, src)
        # Apply transformation
        warped = cv2.warpPerspective(img, matrix, (width, height), flags=cv2.INTER_LINEAR)
        return warped, matrix, inverse_matrix

    # Define a function that thresholds the L-channel of HLS
    # Use exclusive lower bound (>) and inclusive upper (<=)
    def hls_lthresh(self, img, thresh=(220, 255)):
        # 1) Convert to HLS color space
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        hls_l = hls[:,:,1]
        hls_l = hls_l*(255/np.max(hls_l))
        # 2) Apply a threshold to the L channel
        binary_output = np.zeros_like(hls_l)
        binary_output[(hls_l > thresh[0]) & (hls_l <= thresh[1])] = 1
        # 3) Return a binary image of threshold result
        return binary_output

    # Define a function that thresholds the B-channel of LAB
    # Use exclusive lower bound (>) and inclusive upper (<=), OR the results of the thresholds (B channel should capture
    # yellows)
    def lab_bthresh(self, img, thresh=(190,255)):
        # 1) Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
        lab_b = lab[:,:,2]
        # don't normalize if there are no yellows in the image
        if np.max(lab_b) > 175:
            lab_b = lab_b*(255/np.max(lab_b))
        # 2) Apply a threshold to the L channel
        binary_output = np.zeros_like(lab_b)
        binary_output[((lab_b > thresh[0]) & (lab_b <= thresh[1]))] = 1
        # 3) Return a binary image of threshold result
        return binary_output

    def calc_curvature_and_center_dist(self, bin_img, l_fit, r_fit, l_lane_inds, r_lane_inds):
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 3.048/100 # meters per pixel in y dimension, lane line is 10 ft = 3.048 meters
        xm_per_pix = 3.7/378 # meters per pixel in x dimension, lane width is 12 ft = 3.7 meters
        left_curverad, right_curverad, center_dist = (0, 0, 0)
        # Define y-value where we want radius of curvature
        # I'll choose the maximum y-value, corresponding to the bottom of the image
        h = bin_img.shape[0]
        ploty = np.linspace(0, h-1, h)
        y_eval = np.max(ploty)
    
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = bin_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Again, extract left and right line pixel positions
        leftx = nonzerox[l_lane_inds]
        lefty = nonzeroy[l_lane_inds] 
        rightx = nonzerox[r_lane_inds]
        righty = nonzeroy[r_lane_inds]
        
        if len(leftx) != 0 and len(rightx) != 0:
            # Fit new polynomials to x,y in world space
            left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
            right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
            # Calculate the new radii of curvature
            left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
            right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
            # Now our radius of curvature is in meters
        
        # Distance from center is image x midpoint - mean of l_fit and r_fit intercepts 
        if r_fit is not None and l_fit is not None:
            car_position = bin_img.shape[1]/2
            l_fit_x_int = l_fit[0]*h**2 + l_fit[1]*h + l_fit[2]
            r_fit_x_int = r_fit[0]*h**2 + r_fit[1]*h + r_fit[2]
            lane_center_position = (r_fit_x_int + l_fit_x_int) /2
            center_dist = (car_position - lane_center_position) * xm_per_pix
        return round(left_curverad, 1), round(right_curverad, 1), round(center_dist, 1)

    def draw_lane(self, original_img, binary_img, l_fit, r_fit):
        new_img = np.copy(original_img)
        if l_fit is None or r_fit is None:
            return original_img
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(binary_img).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        
        h,w = binary_img.shape
        ploty = np.linspace(0, h-1, num=h)# to cover same y-range as image
        left_fitx = l_fit[0]*ploty**2 + l_fit[1]*ploty + l_fit[2]
        right_fitx = r_fit[0]*ploty**2 + r_fit[1]*ploty + r_fit[2]

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255,0,255), thickness=15)
        cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0,255,255), thickness=15)

        # Warp the blank back to original image space using inverse perspective matrix (inverse_matrix)
        newwarp = cv2.warpPerspective(color_warp, self.inverse_matrix, (w, h)) 
        # Combine the result with the original image
        result = cv2.addWeighted(new_img, 1, newwarp, 0.5, 0)
        return result

    def draw_data(self, original_img, curv_rad, center_dist):
        new_img = np.copy(original_img)
        h = new_img.shape[0]
        font = cv2.FONT_HERSHEY_DUPLEX
        text = 'Curve radius: ' + '{:04.2f}'.format(curv_rad) + 'm'
        cv2.putText(new_img, text, (40,70), font, 1.5, (200,255,155), 2, cv2.LINE_AA)
        direction = ''
        if center_dist > 0:
            direction = 'right'
        elif center_dist < 0:
            direction = 'left'
        abs_center_dist = abs(center_dist)
        text = '{:04.3f}'.format(abs_center_dist) + 'm ' + direction + ' of center'
        cv2.putText(new_img, text, (40,120), font, 1.5, (200,255,155), 2, cv2.LINE_AA)
        return new_img

    # Image pre processing pipeline
    def pre_pipeline(self, image):
        # Undistort Image
        image = self.undistort(image)
        # Perspective Transform
        image, matrix, inverse_matrix = self.unwarp(image, self.src, self.dst)
        self.inverse_matrix = inverse_matrix
        # HLS L-channel Threshold
        image_L = self.hls_lthresh(image)
        # Lab B-channel Threshold
        image_B = self.lab_bthresh(image)
        # Combine HLS and Lab B channel thresholds
        combined = np.zeros_like(image_B)
        combined[(image_L == 1) | (image_B == 1)] = 1

        return combined

    def calibrate_camera(self):
        # Regex: find all calibration files
        input_images = sorted(glob.glob("./assets/camera_cal/calibration*.jpg"))
        print("Calibrating camera with {} images.".format(len(input_images)))
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

        # Calibrate Camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, self.image_size, None, None)
        # record data
        self.mtx = mtx
        self.dist = dist

    # Define method to fit polynomial to binary image with lines extracted, using sliding window
    def sliding_window_polyfit(self, img):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        quarter_point = np.int(midpoint//2)
        # Previously the left/right base was the max of the left/right half of the histogram
        # this changes it so that only a quarter of the histogram (directly to the left/right) is considered
        leftx_base = np.argmax(histogram[quarter_point:midpoint]) + quarter_point
        rightx_base = np.argmax(histogram[midpoint:(midpoint+quarter_point)]) + midpoint

        # Choose the number of sliding windows
        nwindows = 10
        # Set height of windows
        window_height = np.int(img.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 80
        # Set minimum number of pixels found to recenter window
        minpix = 40
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        # Rectangle data for visualization
        rectangle_data = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - (window+1)*window_height
            win_y_high = img.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            rectangle_data.append((win_y_low, win_y_high, win_xleft_low, win_xleft_high, win_xright_low, win_xright_high))
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        left_fit, right_fit = (None, None)
        # Fit a second order polynomial to each
        if len(leftx) != 0:
            left_fit = np.polyfit(lefty, leftx, 2)
        if len(rightx) != 0:
            right_fit = np.polyfit(righty, rightx, 2)
        
        visualization_data = (rectangle_data, histogram)
        
        return left_fit, right_fit, left_lane_inds, right_lane_inds, visualization_data

    # Define method to fit polynomial to binary image based upon a previous fit (chronologically speaking);
    # this assumes that the fit will not change significantly from one video frame to the next
    def polyfit_using_previous_fit(self, binary_warped, left_fit_prev, right_fit_prev):
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 80
        left_lane_inds = ((nonzerox > (left_fit_prev[0]*(nonzeroy**2) + left_fit_prev[1]*nonzeroy + left_fit_prev[2] - margin)) & 
                        (nonzerox < (left_fit_prev[0]*(nonzeroy**2) + left_fit_prev[1]*nonzeroy + left_fit_prev[2] + margin))) 
        right_lane_inds = ((nonzerox > (right_fit_prev[0]*(nonzeroy**2) + right_fit_prev[1]*nonzeroy + right_fit_prev[2] - margin)) & 
                        (nonzerox < (right_fit_prev[0]*(nonzeroy**2) + right_fit_prev[1]*nonzeroy + right_fit_prev[2] + margin)))  

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        left_fit_new, right_fit_new = (None, None)
        if len(leftx) != 0:
            # Fit a second order polynomial to each
            left_fit_new = np.polyfit(lefty, leftx, 2)
        if len(rightx) != 0:
            right_fit_new = np.polyfit(righty, rightx, 2)
        return left_fit_new, right_fit_new, left_lane_inds, right_lane_inds


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = []  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #number of detected pixels
        self.pixel_count = None

    def add_fit(self, fit, inds):
        # add a found fit to the line, up to n
        if fit is not None:
            if self.best_fit is not None:
                # if we have a best fit, see how this new fit compares
                self.diffs = abs(fit-self.best_fit)
            if (self.diffs[0] > 0.001 or \
               self.diffs[1] > 1.0 or \
               self.diffs[2] > 100.) and \
               len(self.current_fit) > 0:
                # bad fit! abort! abort! ... well, unless there are no fits in the current_fit queue, then we'll take it
                self.detected = False
            else:
                self.detected = True
                self.pixel_count = np.count_nonzero(inds)
                self.current_fit.append(fit)
                if len(self.current_fit) > 5:
                    # throw out old fits, keep newest n
                    self.current_fit = self.current_fit[len(self.current_fit)-5:]
                self.best_fit = np.average(self.current_fit, axis=0)
        # or remove one from the history, if not found
        else:
            self.detected = False
            if len(self.current_fit) > 0:
                # throw out oldest fit
                self.current_fit = self.current_fit[:len(self.current_fit)-1]
            if len(self.current_fit) > 0:
                # if there are still any fits in the queue, best_fit is their average
                self.best_fit = np.average(self.current_fit, axis=0)

# instance of helper
helper = Helper()

# instance of left_lane & right_lane
left_lane = Line()
right_lane = Line()
