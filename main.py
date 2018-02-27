# Import Statements
import numpy as np
import cv2, glob, math, pickle
from moviepy.editor import VideoFileClip
from helper import helper, left_lane, right_lane

def pipeline(original_image):
    image = np.copy(original_image)
    bird_view_image = helper.pre_pipeline(image)

    if left_lane.detected and right_lane.detected:
        left_fit, right_fit, left_lane_indices, right_lane_indices = helper.polyfit_using_previous_fit(bird_view_image, left_lane.best_fit, right_lane.best_fit)
    else:
        left_fit, right_fit, left_lane_indices, right_lane_indices, _ = helper.sliding_window_polyfit(bird_view_image)
        
    if left_fit is not None and right_fit is not None:
        height = image.shape[0]
        left_fit_x_intercept = left_fit[0] * height ** 2 + left_fit[1] * height + left_fit[2]
        right_fit_x_intercept = right_fit[0] * height ** 2 + right_fit[1] * height + right_fit[2]
        x_intercept_diff = abs(right_fit_x_intercept - left_fit_x_intercept)
        if abs(350 - x_intercept_diff) > 100:
            left_fit = None
            right_fit = None
        
    left_lane.add_fit(left_fit, left_lane_indices)
    right_lane.add_fit(right_fit, right_lane_indices)

    if left_lane.best_fit is not None and right_lane.best_fit is not None:
        output_image = helper.draw_lane(image, bird_view_image, left_lane.best_fit, right_lane.best_fit)
        left_curvature, right_curvature, distance_from_center = helper.calc_curvature_and_center_dist(bird_view_image, left_lane.best_fit, right_lane.best_fit, left_lane_indices, right_lane_indices)
        output_image = helper.draw_data(output_image, (left_curvature + right_curvature)/2, distance_from_center)
    else:
        output_image = image

    return output_image


# Init Helper
helper.calibrate_camera()

# Process Images
for name in glob.glob("./assets/inputs/*jpg"):
    image = cv2.imread(name)
    output_image = helper.pre_pipeline(image)
    cv2.imwrite(name.split('/')[-1], output_image)

# Process Videos
# for name in glob.glob("./assets/inputs/*mp4"):
#     video = VideoFileClip(name).fl_image(pipeline)
#     video.write_videofile('./outputs/' + name.split('/')[-1], audio=False)
