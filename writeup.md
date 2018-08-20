## Writeup from Marcus Neuert, 2018-08-20

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/1_corners_detected_calibration2.png "Corner detected calibration image 1"
[image2]: ./output_images/1_corners_detected_calibration7.png "Corner detected calibration image 2"
[image3]: ./output_images/2_undistorted_calibration1.png "Applied image correction on calibration image"
[image4]: ./output_images/10_original_image.png "Applied image correction on calibration image"
[image5]: ./output_images/11_undistorted_original_image.png "Road image with region of interest applied"
[image6]: ./output_images/3_image_region_of_interest.png "Road image with region of interest applied"
[image7]: ./output_images/4_undistorted.png "Road image with undistortion"
[image8]: ./output_images/5_warped.png "Warped road image"
[image9]: ./output_images/6_combined_thresholds.png "Threshold applied on warped road image"
[image10]: ./output_images/6_combined_without_color_threshold.png "Bad recognition without color threshold applied"
[image11]: ./output_images/7_sliding_window.png "Sliding window visualization"
[image12]: ./output_images/8_annotate_lane.png "Annotated lane on original image"
[image13]: ./output_images/9_meta_information.png "Annotated lane on original image with meta information"

[video1]: ./output_images/lane_detected_project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is between lines 43 and 108.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. Therefore, I simply used the `cv2.findChessboardCorners` and `cv2.drawChessboardCorners()` methods in my method `get_camera_cali_params()`, to find and draw the detections, as shown here:

![alt text][image1] ![alt text][image2]

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function in my method `calibrate_camera()`.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image3]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction with this image:

![alt text][image4]

Undistorted:
![alt text][image5]

In my preprocessing pipeline, I reduced the image to a triangle of interest.
Code in line 218 through 250.
Definition of my triangle:

````sh
vertices = np.array([[
        (MIN_LANE_OFFSET, img.shape[0] - 50),
        (img.shape[1] / 2 - 50, img.shape[0] / 2 + 50),
        (img.shape[1] / 2 + 50, img.shape[0] / 2 + 50),
        (img.shape[1] - MIN_LANE_OFFSET, img.shape[0] - 50)]],
        dtype=np.int32)
````

Which gives these points:

| Vertices     | 
:-------------:| 
| 100, 670     | 
| 590, 410     |
| 690, 410     |
| 1180, 670    |

And looks like this, first:

![alt text][image6]

And with applied undistortion:
![alt text][image7]

#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

After the region of interest cut-out, I did the perspective transform.

The code for my perspective transform includes a function called `unwarp()`, which appears in lines 254 through 261   The `warper()` function takes as inputs an image (`img`). It also requires the global variables source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner within the `do_calibration()` method in lines 97 to 106:

```sh
    src_points = np.float32([(540, 488),
                             (750, 488),
                             (777, 508),
                             (507, 508)])

    dst_points = np.float32([(600, 300),
                             (700, 300),
                             (700, 420),
                             (600, 420)])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 540, 488      | 600, 300      | 
| 750, 488      | 700, 300      |
| 777, 508      | 700, 420      |
| 507, 508      | 600, 420      |

Here is the result:

![alt text][image8]

#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 111 through 215).  I first took out the yellow and white colors with `color_threshold` and then applied the Sobel, Magnitude and Direction of the Gradient thresholds. See why I used color threshold on white and yellow in the Discussions section.

Combined Thresholds only:
![alt text][image10]

Applied color threshold, too:
![alt text][image9]


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image11]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 413 through 451 in my code in `measure_curvature_and_center_distance()`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result:

![alt text][image12]

Annotated lane with information about curve radius and distance from center of the lane:
![alt text][image13]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_images/lane_detected_project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I faced a lot of problems with shadows on the left side, so that my lane was plotted on the wall or between the wall and the lane. I quickly used the region of interest code from project one, but it failed a long time. The reason was, that I applied the region of interest filter too late or without color thresholding. Thereby, the frame of the triangle was recognized as another lane, as you can see in the following image:

![alt text][image10]

I also had some problem with finding the correct destination array for the image, as I sticked to hard to the values from the lessons. I finally came up with a very narrow, but long road.

Another problem was the calculation of the curvature. I did a long debugging session until i found my bugs in recalculation the nonzero's from the lane_inds' instead of reusing the values from the previous method.
