
## Advance Lane Finding Algorithm

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

[image1]: ./output_images/cam_corners/calibration3.jpg " CheckerBoard Corners"
[image2]: ./output_images/undistorted/test4.jpg "Undistorted images"
[image3]: ./output_images/binary_images/test2.jpg "Binary Example"
[image4]: ./output_images/perspective_top/source_unwarped.jpg "Warp Example"

[image5]: ./output_images/perspective_top/source_warped.jpg "Warp Example"
[image6]:./output_images/perspective_top/source_inverse_perspective_wraped.jpg "Warped Example "
[image7]: ./output_images/curve_fitting/selected_boxes.jpg "Fit Visual"

[image8]: ./output_images/curve_fitting/lane_curve_fitting.jpg "Fit Visual"
[image9]: ./output_images/final/final.jpg "Output"
[video1]: ./project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained  "./cam_caliberate.py".  

I started by extracting the corner points for each checkerboard image. I wrote a small function called `caliberate` and used opencv `findChessboardCorners` function to find all corners.Here is the example output

![all text][image1] 


 I generated output images after finding corners in the folder `./output_images/cam_corners`

I saved the image points and object points in `./output_images/cam_corners/img_wld.pickle`. I will be using this as reference for all my images or videos.
 

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

 The code for this step is contained  `./cam_caliberate.py`. 


 After caliberation of camera, now I have 2D to 3D point conversion reference points in `./output_images/cam_corners/img_wld.pickle` . I can use this to find the ammount of distortion present in the camera using opencv `cv2.calibrateCamera`  which returns the distortion co-efficents and camera matrix. Once I have these paramters I can use  opencv `cv2.undistort` function to correct it.
 
  I used the camera matrix and distortion co-efficients to correct the distorion
 ![alt text][image2]

  All the code for this is captured in `undistort` function. Also there is `save_distortion_correction` function which saves the output for these `test_images`. My output images are present in this `./output_images/undistorted`  foder.


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

All the code for this step is present in `./gradient.py`

I wrote a funtion called `get_binary` which takes a RBG image and generates binary thresholded image. I used combination of color thresholding using HSV space, magnitude thesholding using Sobel kernel to get the binary output image. 


Here is the sample output

![alt text][image3]

My output images are present in this `./output_images/binary_images/`  foder.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

All the code for this step is present in the 
`./ALF.py`
The code for my perspective transform is in a function called `get_top_view()`, which is present in the file `./ALF.py`. The `get_top_view()` function takes as input an single channel image and returns the top-view for that using opencv `cv2.warpPerspective` also returns the perpective matrices `M,Minv` generated using`cv2.getPerspectiveTransform`.  I chose the source and destination points using trial and error to get the best fit.

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.
#### Source image with trapezoid marked as source

![alt text][image4]

#### Top_view of the source image

![alt text][image5]

#### Inverse perspective image
![alt text][image6]

Example out put for `test_images` are present in `./output_images/top_view_binary` and `./output_images/top_view` folders respectievly.

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code corresponding to this step is contained in function `detect_lane` found as a member function of `LTracker` object present in the `./Lane.py` file 

I used windows search using the histogram method to detect lane position. Here the base position of lane is calculated as the histogram of the binary warped image. The peak of the histogram indicates the possible lane positions and we search arround the peak to find the non-zeros pixels.Each search is subdivided into boxes so as to get better search results.

Here is the windows search results clearly indicates the detected lane pixels
![alt text][image7]

Using these pixel locations we fit the second order polynomial using `np.polyfit`.

![alt text][image8]


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The code corresponding to this step is contained in function `get_radius_of_curvature` found as a member function of `LTracker` object present in the `./Lane.py` file. I used following conversion to measure real world curvature i.e described in here
```
ym_per_pix = 30 / shape[0]  # meters per pixel in y dimension
xm_per_pix = 3.7 /shape[1] # meters per pixel in x dimension
```

 Radius of curvature is computed using the standard formula described in [here](https://www.intmath.com/applications-differentiation/8-radius-curvature.php)

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I used the tracker to generate the lane overlay image and used inversepective to draw the ovelay ontop of the original image.
The code is present in `process_image` function in `./ALF.py` file.

![alt text][image9]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust? 

### Lane Detection
#### Good things about current algorithm
1. The current lane detection is good in highways where in good lane marking can be found.
2. It also takes care of yellow lane if any.
3. Also has a simple moving average tracker to track the lane in case of miss detections. I am also running the detector every 20 frames so we get better curve fit.
####  Possible imporvements
1. The current algorthm works well provided there is no shadow changes. No care is taken in case if there is no detection.
2. What I found during the my experimentation is the current algorithm fails when there are sharper turns given in `harder_challenge_video.mp4`. So we could use small trapezoid to find the perspective transfrom.
3. When there is no lane all the features all lost in the image, in such cases we could use some dynamic thresholding based on if it is a shadow region or very bright region.
4. Aslo in the code there is no constraint added that detected lanes should be parallel which makes the algo very fragile for sudden changes in car steering angle.
