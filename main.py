import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
from moviepy.editor import VideoFileClip

CALIBRATION__JPG = "calibration1.jpg"
TRANSFORMATION__JPG = "test6.jpg"
TRANSFORMATION2__JPG = "test4.jpg"
FOLDER_TEST_INPUT = "test_images/"
FOLDER_OUTPUT = "output_images/"
FOLDER_CAMERA_CAL = "camera_cal/"
FOLDER_VIDEOS_INPUT = "input_videos/"
INPUT_VIDEO = "project_video.mp4"
#INPUT_VIDEO = "challenge_video.mp4"
# INPUT_VIDEO = "harder_challenge_video.mp4"
OUTPUT_VIDEO = "lane_detected_video.mp4"

# debug controls
TEST_RUN = True
VIDEO_LENGTH_SECONDS = 1

# HYPERPARAMETERS
SOBEL_KERNEL = 7
# Choose the number of sliding windows
N_WINDOWS = 10
# Set the width of the windows +/- margin
WINDOW_MARGIN = 20
# Set minimum number of pixels found to recenter window
WINDOW_RECENTER_MIN_PIX = 40
# Choose the width of the margin around the previous polynomial to search
NEW_LANE_MARGIN = 80
# minimum distance of lane from image borders
MIN_LANE_OFFSET = 150


# Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
def get_camera_cali_params():
    nx, ny = 6, 9
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(FOLDER_CAMERA_CAL + 'calibration*.jpg')

    # Step through the list and search for chessboard corners
    for fname in images:

        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, add object points, image points
        # if not ret:
        #     nx, ny = 9,5
        #     objp = np.zeros((nx * ny, 3), np.float32)
        #     objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
        #     ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        # if not ret:
        #     nx, ny = 8,6
        #     objp = np.zeros((nx * ny, 3), np.float32)
        #     objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
        #     ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            if TEST_RUN:
                # Draw and display the corners
                img_with_corners = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
                save_image_as_png(img_with_corners, "1_corners_detected_", fname.split('\\')[-1].split('.')[0])
        else:
            print("No corners found in image: " + fname)
    return objpoints, imgpoints


# save image as png to OUTPUT_FOLDER
def save_image_as_png(img, prefix, filename):
    cv2.imwrite(FOLDER_OUTPUT + prefix + filename + ".png", img)


# get coefficients from camera calibration
def calibrate_camera(folder, img, obj_p, img_p):
    filename = img.split('.')[0]
    img = cv2.imread(folder + img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_p, img_p, gray.shape[0:2], None, None)

    if TEST_RUN:
        dst = cv2.undistort(img, mtx, dist, None, mtx)
        save_image_as_png(dst, "2_undistorted_", filename)

    return mtx, dist


def do_calibration():
    obj_points, img_points = get_camera_cali_params()
    cali_mtx, cali_dist = calibrate_camera(FOLDER_CAMERA_CAL, CALIBRATION__JPG, obj_points, img_points)
    calibration_file = FOLDER_CAMERA_CAL + CALIBRATION__JPG
    calibration_file_undistorted = cv2.undistort(cv2.imread(calibration_file), cali_mtx, cali_dist, None, cali_mtx)
    h, w = calibration_file_undistorted.shape[:2]
    offset = 450
    # define source and destination points for transform
    src = np.float32([(575, 464),
                      (707, 464),
                      (258, 682),
                      (1049, 682)])
    dst = np.float32([(offset, 0),
                      (w - offset, 0),
                      (offset, h),
                      (w - offset, h)])
    return cali_mtx, cali_dist, h, src, dst


# from Udacity
def s_channel_and_gradient_threshold(img):
    # Convert to HLS color space and separate the S channel
    # Note: img is the undistorted image
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]

    # Grayscale image
    # NOTE: we already saw that standard grayscaling lost color information for the lane lines
    # Explore gradients in other colors spaces / color channels to see what might work better
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=SOBEL_KERNEL)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    thresh_min = 20
    thresh_max = 90
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Threshold color channel
    s_thresh_min = 180
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    # color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    return combined_binary


# Udacity_project_1
def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)

    return masked_image


# from Udacity
def unwarp(img, src, dst):
    h, w = img.shape[:2]
    # use cv2.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)
    return warped, M, Minv


def preprocess_pipeline(img):
    undist_image = cv2.undistort(img, cali_mtx, cali_dist, None, cali_mtx)

    c_binary = s_channel_and_gradient_threshold(undist_image)

    imshape = c_binary.shape
    vertices = np.array([[
        (150, imshape[0]),
        (imshape[1] / 2, 400),
        (imshape[1] / 2, 400),
        (imshape[1] - 150, imshape[0])]],
        dtype=np.int32)

    image_region_of_interest = region_of_interest(c_binary, vertices)

    warped_bin, M, Minv = unwarp(image_region_of_interest, src, dst)

    # return warped_binary, M, Minv
    return warped_bin, Minv


def histogram(img):
    # Grab only the bottom half of the image
    # Lane lines are likely to be mostly vertical nearest to the car
    bottom_half = img[img.shape[0] // 2:, :]

    # Sum across image pixels vertically - make sure to set an `axis`
    # i.e. the highest areas of vertical lines should be larger values
    histogram = np.sum(bottom_half, axis=0)

    return histogram


def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    hist = histogram(binary_warped)

    # Create an output image to draw on and visualize the result
    rectangle_data = []

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(hist.shape[0] // 2)
    leftx_base = np.argmax(hist[:midpoint])
    rightx_base = np.argmax(hist[midpoint:]) + midpoint

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // N_WINDOWS)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(N_WINDOWS):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - WINDOW_MARGIN
        win_xleft_high = leftx_current + WINDOW_MARGIN
        win_xright_low = rightx_current - WINDOW_MARGIN
        win_xright_high = rightx_current + WINDOW_MARGIN

        rectangle_data.append((win_y_low, win_y_high, win_xleft_low, win_xleft_high, win_xright_low, win_xright_high))

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > WINDOW_RECENTER_MIN_PIX:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > WINDOW_RECENTER_MIN_PIX:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit, right_fit = (None, None)
    if len(leftx) != 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit, left_lane_inds, right_lane_inds, rectangle_data, hist


def fit_polynomal(binary_warped, left_fit, right_fit):
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    try:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    return left_fitx, right_fitx, ploty


def search_around_poly(binary_warped, left_fit, right_fit):
    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = (
            (nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - NEW_LANE_MARGIN)) &
            (nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + NEW_LANE_MARGIN)))
    right_lane_inds = ((nonzerox > (
            right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - NEW_LANE_MARGIN)) &
                       (nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[
                           2] + NEW_LANE_MARGIN)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # left_fitx, right_fitx, left_lane_inds, right_lane_inds, ploty = find_lane_pixels(binary_warped)

    # ## Visualization ##
    # # Create an image to draw on and an image to show the selection window
    # out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # window_img = np.zeros_like(out_img)
    # # Color in left and right line pixels
    # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    #
    # # Generate a polygon to illustrate the search window area
    # # And recast the x and y points into usable format for cv2.fillPoly()
    # left_line_window1 = np.array([np.transpose(np.vstack([leftx - NEW_LANE_MARGIN, ploty]))])
    # left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([leftx + NEW_LANE_MARGIN, ploty])))])
    # left_line_pts = np.hstack((left_line_window1, left_line_window2))
    # right_line_window1 = np.array([np.transpose(np.vstack([rightx - NEW_LANE_MARGIN, ploty]))])
    # right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([rightx + NEW_LANE_MARGIN, ploty])))])
    # right_line_pts = np.hstack((right_line_window1, right_line_window2))
    #
    # # Draw the lane onto the warped blank image
    # cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    # cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    # result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    ## End visualization steps ##

    # return result, left_lane_inds, right_lane_inds, ploty

    left_fit_new, right_fit_new = (None, None)
    if len(leftx) != 0:
        # Fit a second order polynomial to each
        left_fit_new = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit_new = np.polyfit(righty, rightx, 2)

    return left_fit_new, right_fit_new, left_lane_inds, right_lane_inds


def measure_curvature_and_center_distance(bin_img, l_fit, r_fit, l_lane_inds, r_lane_inds):
    # Calculates the curvature of polynomial functions in pixels.

    # Define conversions in x and y from pixels space to meters
    width = dst[3][1] - dst[0][1]
    height = dst[1][0] - dst[0][0]

    # TODO: Check ft vs. meters
    ym_per_pix = 3.048/100 # meters per pixel in y dimension, lane line is 10 ft = 3.048 meters
    xm_per_pix = 3.7/378 # meters per pixel in x dimension, lane width is 12 ft = 3.7 meters
    #ym_per_pix = 12.0 / height  # meters per pixel in y dimension
    #xm_per_pix = 10.0 / width  # meters per pixel in x dimension

    ploty = np.linspace(0, bin_img.shape[0] - 1, bin_img.shape[0])

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty) * ym_per_pix

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = bin_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Again, extract left and right line pixel positions
    leftx = nonzerox[l_lane_inds]
    lefty = nonzeroy[l_lane_inds]
    rightx = nonzerox[r_lane_inds]
    righty = nonzeroy[r_lane_inds]

    left_curverad, right_curverad, distance_from_center = (0, 0, 0)

    if len(leftx) != 0 and len(rightx) != 0:
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = ((1 + (
                2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])

    if len(l_fit) == 3 and len(r_fit) == 3:
        car_position = bin_img.shape[1] / 2
        l_fit_x_int = l_fit[0] * h ** 2 + l_fit[1] * h + l_fit[2]
        r_fit_x_int = r_fit[0] * h ** 2 + r_fit[1] * h + r_fit[2]
        lane_center_position = (r_fit_x_int + l_fit_x_int) / 2
        distance_from_center = (car_position - lane_center_position) * xm_per_pix

    return left_curverad, right_curverad, distance_from_center


# TODO: from Udacity project lines on original image
def draw_lane(original_img, binary_img, l_fit, r_fit, Minv):
    new_img = np.copy(original_img)

    if l_fit is None or len(l_fit) != 3 or r_fit is None or len(r_fit) != 3:
        return original_img

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    h, w = binary_img.shape[:2]
    ploty = np.linspace(0, h - 1, num=h)
    left_fitx = l_fit[0] * ploty ** 2 + l_fit[1] * ploty + l_fit[2]
    right_fitx = r_fit[0] * ploty ** 2 + r_fit[1] * ploty + r_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    #cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255, 0, 255), thickness=15)
    #cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0, 255, 255), thickness=15)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (w, h))
    # Combine the result with the original image
    result = cv2.addWeighted(new_img, 1, newwarp, 0.5, 0)
    return result


def draw_data(original_img, curv_rad, center_dist):
    new_img = np.copy(original_img)

    font = cv2.FONT_ITALIC
    text = 'Curve radius: ' + '{:04.2f}'.format(curv_rad) + 'm'
    cv2.putText(new_img, text, (40, 70), font, 1.5, (200, 255, 155), 2, cv2.LINE_AA)

    direction = ''
    if center_dist > 0:
        direction = 'right'
    elif center_dist < 0:
        direction = 'left'
    abs_center_dist = abs(center_dist)

    text = '{:04.3f}'.format(abs_center_dist) + 'm ' + direction + ' of center'
    cv2.putText(new_img, text, (40, 120), font, 1.5, (200, 255, 155), 2, cv2.LINE_AA)

    return new_img


# video pipeline
def process_image(img):
    new_img = np.copy(img)
    img_bin, Minv = preprocess_pipeline(new_img)

    # if both left and right lines were detected last frame, use polyfit_using_prev_fit, otherwise use sliding window
    if not l_line.detected or not r_line.detected:
        l_fit, r_fit, l_lane_inds, r_lane_inds, rectangle, hist = find_lane_pixels(img_bin)
    else:
        l_fit, r_fit, l_lane_inds, r_lane_inds = search_around_poly(img_bin, l_line.best_fit, r_line.best_fit)

    # invalidate both fits if the difference in their x-intercepts isn't around 350 px (+/- 100 px)
    if l_fit is not None and r_fit is not None:
        # calculate x-intercept (bottom of image, x=image_height) for fits
        h = img.shape[0]
        l_fit_x_int = l_fit[0] * h ** 2 + l_fit[1] * h + l_fit[2]
        r_fit_x_int = r_fit[0] * h ** 2 + r_fit[1] * h + r_fit[2]
        x_int_diff = abs(r_fit_x_int - l_fit_x_int)
        if abs(350 - x_int_diff) > 100:
            l_fit = None
            r_fit = None

    l_line.add_fit(l_fit, l_lane_inds)
    r_line.add_fit(r_fit, r_lane_inds)

    # draw the current best fit if it exists
    if l_line.best_fit is not None and r_line.best_fit is not None:
        output_image = draw_lane(new_img, img_bin, l_line.best_fit, r_line.best_fit, Minv)
        rad_l, rad_r, d_center = measure_curvature_and_center_distance(img_bin, l_line.best_fit, r_line.best_fit,
                                                                       l_lane_inds, r_lane_inds)
        output_image = draw_data(output_image, (rad_l + rad_r) / 2, d_center)
    else:
        output_image = new_img

    return output_image


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

    def add_fit(self, fit, inds):
        # add a found fit to the line, up to n
        if fit is not None:
            if self.best_fit is not None:
                # if we have a best fit, see how this new fit compares
                self.diffs = abs(fit - self.best_fit)
            if (self.diffs[0] > 0.001 or self.diffs[1] > 1.0 or self.diffs[2] > 100.) and len(self.current_fit) > 0:
                # bad fit! abort! abort! ... well, unless there are no fits in the current_fit queue, then we'll take it
                self.detected = False
            else:
                self.detected = True
                self.px_count = np.count_nonzero(inds)
                self.current_fit.append(fit)
                if len(self.current_fit) > 5:
                    # throw out old fits, keep newest n
                    self.current_fit = self.current_fit[len(self.current_fit) - 5:]
                self.best_fit = np.average(self.current_fit, axis=0)
        # or remove one from the history, if not found
        else:
            self.detected = False
            if len(self.current_fit) > 0:
                # throw out oldest fit
                self.current_fit = self.current_fit[:len(self.current_fit) - 1]
            if len(self.current_fit) > 0:
                # if there are still any fits in the queue, best_fit is their average
                self.best_fit = np.average(self.current_fit, axis=0)


def write_video():
    video_output = FOLDER_OUTPUT + OUTPUT_VIDEO

    # To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    if TEST_RUN:
        input_clip = VideoFileClip(FOLDER_VIDEOS_INPUT + INPUT_VIDEO).subclip(0, VIDEO_LENGTH_SECONDS)
    else:
        input_clip = VideoFileClip(FOLDER_VIDEOS_INPUT + INPUT_VIDEO)

    output_clip = input_clip.fl_image(process_image)
    output_clip.write_videofile(video_output, audio=False)


# TODO: import video as array of images
# TODO: Apply a distortion correction to raw images.
# TODO: Use color transforms, gradients, etc., to create a thresholded binary image.
# TODO: Apply a perspective transform to rectify binary image ("birds-eye view").
# TODO: Detect lane pixels and fit to find the lane boundary.
# TODO: Determine the curvature of the lane and vehicle position with respect to center.
# TODO: Warp the detected lane boundaries back onto the original image.
# TODO: Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

cali_mtx, cali_dist, h, src, dst = do_calibration()
l_line = Line()
r_line = Line()
write_video()

print("Finished")
