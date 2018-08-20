import glob
import cv2
import numpy as np
from moviepy.editor import VideoFileClip

CALIBRATION__JPG = "calibration1.jpg"
FOLDER_OUTPUT = "output_images/"
FOLDER_CAMERA_CAL = "camera_cal/"
FOLDER_VIDEOS_INPUT = "input_videos/"
INPUT_VIDEO = "project_video.mp4"
# INPUT_VIDEO = "challenge_video.mp4"
# INPUT_VIDEO = "harder_challenge_video.mp4"

# debug controls
TEST_RUN = False
VISUALIZATION = False
VIDEO_LENGTH_SUB = (10, 11)

# preprocessing techniques
REGION_OF_INTEREST = True
COLOR_THRESHOLD = True  # without focus on white and yellow, the frame of interested region is misunderstood
S_CHANNEL = True

# HYPERPARAMETERS
SOBEL_KERNEL = 7
# Choose the number of sliding windows
N_WINDOWS = 10
# Set the width of the windows +/- margin
WINDOW_MARGIN = 20
# Set minimum number of pixels found to recenter window
WINDOW_RECENTER_MIN_PIX = 5
# Choose the width of the margin around the previous polynomial to search
NEW_LANE_MARGIN = 80
# minimum distance of lane from image borders
MIN_LANE_OFFSET = 100


# save image as png to OUTPUT_FOLDER
def save_image_as_png(img, prefix, filename):
    cv2.imwrite(FOLDER_OUTPUT + prefix + filename + ".png", img)


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

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            if VISUALIZATION:
                # Draw and display the corners
                img_with_corners = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
                save_image_as_png(img_with_corners, "1_corners_detected_", fname.split('\\')[-1].split('.')[0])
        else:
            print("No corners found in image: " + fname)
    return objpoints, imgpoints


# get coefficients from camera calibration
def calibrate_camera(folder, img, obj_p, img_p):
    img = cv2.imread(folder + img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_p, img_p, gray.shape[0:2], None, None)

    return mtx, dist


# camera calibration pipeline
def do_calibration():
    obj_points, img_points = get_camera_cali_params()
    camera_mtx, camera_dist = calibrate_camera(FOLDER_CAMERA_CAL, CALIBRATION__JPG, obj_points, img_points)

    calibration_file = FOLDER_CAMERA_CAL + CALIBRATION__JPG
    if VISUALIZATION:
        undistorted_image = cv2.undistort(cv2.imread(calibration_file), camera_mtx, camera_dist, None, camera_mtx)
        save_image_as_png(undistorted_image, "2_undistorted_", CALIBRATION__JPG.split(".")[0])

    # define source and destination points for transform
    src_points = np.float32([(540, 488),
                             (750, 488),
                             (777, 508),
                             (507, 508)])

    dst_points = np.float32([(600, 300),
                             (700, 300),
                             (700, 420),
                             (600, 420)])

    return camera_mtx, camera_dist, src_points, dst_points


# filter yellow and white color
def color_threshold(image):
    """
    Removes pixels that are not within the color ranges
    :param image:
    :return:
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    white_mask = cv2.inRange(
        image,
        lowerb=np.array([int(0.0 * 255), int(0.0 * 255), int(0.80 * 255)], dtype="uint8"),
        upperb=np.array([int(1.0 * 255), int(0.10 * 255), int(1.0 * 255)], dtype="uint8")
    )

    yellow_mask = cv2.inRange(
        image,
        lowerb=np.array([int(0.2 * 255), int(0.3 * 255), int(0.10 * 255)], dtype="uint8"),
        upperb=np.array([int(0.6 * 255), int(0.8 * 255), int(0.90 * 255)], dtype="uint8")
    )
    image = cv2.bitwise_or(
        white_mask,
        yellow_mask
    )

    return image


# Apply Sobel, Magnitude & Direction of the Gradient
def combined_thresholds(image):
    """
    Using the Sobel filter, find the edge pixels for the lane lines
    :param image:
    :return:
    """

    # Define a function that takes an image, gradient orientation,
    # and threshold min / max values.
    def abs_sobel_thresh(img, orient='x', thresh=None, sobel_kernel=None):
        thresh_min = thresh[0]
        thresh_max = thresh[1]
        # Apply x or y gradient with the OpenCV Sobel() function
        # and take the absolute value
        # Convert to grayscale
        gray = img
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
        elif orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
        else:
            raise Exception('Invalid `orient`')
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        # Create a copy and apply the threshold
        binary_output = np.zeros_like(scaled_sobel)
        # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
        binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

        # Return the result
        return binary_output

    # Define a function to return the magnitude of the gradient
    # for a given sobel kernel size and threshold values
    def mag_threshold(img, sobel_kernel=3, mag_thresh=(0, 255)):
        # Convert to grayscale
        gray = img
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag) / 255
        gradmag = (gradmag / scale_factor).astype(np.uint8)
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

        # Return the binary image
        return binary_output

    def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
        # Grayscale
        # Calculate the x and y gradients
        gray = img
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Take the absolute value of the gradient direction,
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output = np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

        # Return the binary image
        return binary_output

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=SOBEL_KERNEL, thresh=(0, 30))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=SOBEL_KERNEL, thresh=(20, 90))
    mag_binary = mag_threshold(image, sobel_kernel=SOBEL_KERNEL, mag_thresh=(0, 10))
    dir_binary = dir_threshold(image, sobel_kernel=SOBEL_KERNEL, thresh=(0, np.pi / 4))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 0) & (grady == 0)) | ((mag_binary == 0) & (dir_binary == 0))] = 255

    return combined


# filter on region of interest - from Udacity_project_1
def region_of_interest(img):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """

    vertices = np.array([[
        (MIN_LANE_OFFSET, img.shape[0] - 50),
        (img.shape[1] / 2 - 50, img.shape[0] / 2 + 50),
        (img.shape[1] / 2 + 50, img.shape[0] / 2 + 50),
        (img.shape[1] - MIN_LANE_OFFSET, img.shape[0] - 50)]],
        dtype=np.int32)

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


# simple perspective transformation
def unwarp(img):
    h, w = img.shape[:2]
    # use cv2.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)
    return warped, M, Minv


# image preprocessing pipeline
def preprocess_pipeline(img):
    # load image and convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if VISUALIZATION:
        save_image_as_png(img, "10_", "original_image")
        save_image_as_png(cv2.undistort(img, cali_mtx, cali_dist, None, cali_mtx), "11_", "undistorted_original_image")

    # Apply region of interest filter on image, if enabled
    if REGION_OF_INTEREST:
        img = region_of_interest(img)
        if VISUALIZATION:
            save_image_as_png(img, "3_", "image_region_of_interest")

    # Apply a distortion correction to raw images.
    img = cv2.undistort(img, cali_mtx, cali_dist, None, cali_mtx)
    if VISUALIZATION:
        save_image_as_png(img, "4_", "undistorted")

    # Apply a perspective transform to rectify binary image ("birds-eye view").
    img, M, Minv = unwarp(img)
    if VISUALIZATION:
        save_image_as_png(img, "5_", "warped")
        save_image_as_png(combined_thresholds(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)), "6_",
                          "combined_without_color_threshold")

    # filter yellow and white color from image, if enabled
    if COLOR_THRESHOLD:
        img = color_threshold(img)
    else:
        # Convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply color thresholds on image
    img = combined_thresholds(img)
    if VISUALIZATION:
        save_image_as_png(img, "6_", "combined_thresholds")

    return img, Minv


# Detect lane pixels and fit to find the lane boundary.
def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    half = int(binary_warped.shape[0] / 2)
    histogram = np.sum(binary_warped[half:, :], axis=0)

    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] / N_WINDOWS)
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

        if VISUALIZATION:
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
            if window == N_WINDOWS - 1:
                save_image_as_png(out_img, "7_", "sliding_window")

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

    return left_fit, right_fit, left_lane_inds, right_lane_inds, leftx, rightx, lefty, righty


def search_around_poly(binary_warped, left_fit, right_fit):
    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Set the area of search based on activated x-values within the +/- margin of our polynomial function
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

    # return result, left_lane_inds, right_lane_inds, ploty

    left_fit_new, right_fit_new = (None, None)
    if len(leftx) != 0:
        # Fit a second order polynomial to each
        left_fit_new = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit_new = np.polyfit(righty, rightx, 2)

    return left_fit_new, right_fit_new, left_lane_inds, right_lane_inds, leftx, rightx, lefty, righty


# Determine the curvature of the lane and vehicle position with respect to center.
def measure_curvature_and_center_distance(bin_img, l_fit, r_fit, leftx, rightx, lefty, righty):
    # Calculates the curvature of polynomial functions in pixels.

    # Define conversions in x and y from pixels space to meters
    width = 100
    height = bin_img.shape[0]

    ym_per_pix = 30 / height  # meters per pixel in y dimension
    xm_per_pix = 3.7 / width  # meters per pixel in x dimension

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = height

    left_curverad, right_curverad, distance_from_center = 0, 0, 0

    # calculate radius
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

    # calculate distance from center
    if len(l_fit) == 3 and len(r_fit) == 3:
        car_position = bin_img.shape[1] / 2
        l_fit_x_int = l_fit[0] * height ** 2 + l_fit[1] * height + l_fit[2]
        r_fit_x_int = r_fit[0] * height ** 2 + r_fit[1] * height + r_fit[2]
        lane_center_position = (r_fit_x_int + l_fit_x_int) / 2
        distance_from_center = (car_position - lane_center_position) * xm_per_pix

    radius = ((left_curverad + right_curverad) / 2.)
    return radius, distance_from_center


# Warp the detected lane boundaries back onto the original image.
def draw_lane(original_img, warped_img, l_fit, r_fit, Minv):
    # new_img = np.copy(original_img)

    if l_fit is None or r_fit is None:
        # if l_fit is None or len(l_fit) != 3 or r_fit is None or len(r_fit) != 3:
        return original_img

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    h, w = original_img.shape[:2]
    ploty = np.linspace(0, h - 1, num=h)
    left_fitx = l_fit[0] * ploty ** 2 + l_fit[1] * ploty + l_fit[2]
    right_fitx = r_fit[0] * ploty ** 2 + r_fit[1] * ploty + r_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255, 0, 255), thickness=5)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0, 255, 255), thickness=5)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (w, h))
    # Combine the result with the original image
    result = cv2.addWeighted(original_img, 1, newwarp, 0.3, 0)
    return result


# Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
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
        l_fit, r_fit, l_lane_inds, r_lane_inds, leftx, rightx, lefty, righty = find_lane_pixels(img_bin)
    else:
        l_fit, r_fit, l_lane_inds, r_lane_inds, leftx, rightx, lefty, righty = search_around_poly(img_bin,
                                                                                                  l_line.best_fit,
                                                                                                  r_line.best_fit)

    l_line.add_fit(l_fit, l_lane_inds)
    r_line.add_fit(r_fit, r_lane_inds)

    # draw the current best fit if it exists
    if l_fit is not None and r_fit is not None:
        output_image = draw_lane(new_img, img_bin, l_fit, r_fit, Minv)
        if VISUALIZATION:
            save_image_as_png(cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR), "8_", "annotate_lane")

        radius, d_center = measure_curvature_and_center_distance(new_img, l_fit, r_fit, leftx, rightx, lefty, righty)
        output_image = draw_data(output_image, radius, d_center)
        if VISUALIZATION:
            save_image_as_png(cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR), "9_", "meta_information")
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
            # if len(self.current_fit) > 0:
            # if there are still any fits in the queue, best_fit is their average
            # self.best_fit = np.average(self.current_fit, axis=0)


# import video, process images, write image
def write_video():
    video_output = FOLDER_OUTPUT + "lane_detected_" + INPUT_VIDEO

    # To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    if TEST_RUN:
        input_clip = VideoFileClip(FOLDER_VIDEOS_INPUT + INPUT_VIDEO).subclip(VIDEO_LENGTH_SUB[0], VIDEO_LENGTH_SUB[1])
    else:
        input_clip = VideoFileClip(FOLDER_VIDEOS_INPUT + INPUT_VIDEO)

    output_clip = input_clip.fl_image(process_image)
    output_clip.write_videofile(video_output, audio=False)


# Start lane line recognition with camera calibration
cali_mtx, cali_dist, src, dst = do_calibration()
l_line = Line()
r_line = Line()
write_video()

print("Finished")
