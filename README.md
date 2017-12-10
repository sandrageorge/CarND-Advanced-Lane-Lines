## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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
[image11]: ./writeup_images/chess.jpg
[image12]: ./writeup_images/chess_undist_test_image.jpg
[image1]: ./writeup_images/undist_img.jpg "Undistorted"
[image2]: ./test_images/test2.jpg "Road Transformed"
[image3]: ./writeup_images/binary.jpg "Binary Example"
[image6]: ./writeup_images/perspective_img.jpg "Binary Example"
[image7]: ./writeup_images/out_img.jpg "Binary Example"
[image4]: ./writeup_images/out_img.jp "Fit Visual"
[image5]: ./writeup_images/result.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/sandrageorge/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Code existing in advancedLines.py, lines (29-40)

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image11]  ![alt text][image12] 

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image1]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 49 through 83 in `advancedLines.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)
(Mask is generated for different colors -White and Yellow, then thresholding is done over grayscale image)
![alt text][image2]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspective_transform()`, which appears in lines 82 through 114 in the file `advancedLines.py`.  The `perspective_transform()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
    if src_in is None:
        src = np.array([[600. / 1280. * img_size[1], 455. / 720. * img_size[0]],
                        [690. / 1280. * img_size[1], 455. / 720. * img_size[0]],
                        [1100. / 1280. * img_size[1], 720. / 720. * img_size[0]],
                        [210. / 1280. * img_size[1], 720. / 720. * img_size[0]]], np.float32)

    else:
        src = src_in

    if dst_in is None:
        dst = np.array([[320. / 1280. * img_size[1], 100. / 720. * img_size[0]],
                        [960. / 1280. * img_size[1], 100. / 720. * img_size[0]],
                        [960. / 1280. * img_size[1], 720. / 720. * img_size[0]],
                        [320. / 1280. * img_size[1], 720. / 720. * img_size[0]]], np.float32)
    else:
        dst = dst_in
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 600, 455      | 320, 100      | 
| 690, 455      | 960, 100      |
| 1100, 720     | 960, 720      |
| 210, 720      | 320, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image3]

The prespective image 

![alt text][image6]

Output of finding lane function 

![alt text][image7]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

After undistorting the image, extracting binary data, and applying perspective transform, the next required step is to extract pixels that are associated to lane lines. This was done using the sliding window search. 

![alt text][image4]

Then I fit my detected lane pixels with a 2nd order polynomial 

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 275 through 293 in my code in `advancedLines.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

![alt text][image5]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

Challenges
1- Binary data extraction: I used Sobel X with a threshold X-channel from HSV colour space. Then i migrated to the white mask from RGB space and yellow mask from HSV, these perfectly detected lane lines.
2- Finding a way to implement the lane update function instead of executing a blind search for every frame. 
The Line() helped to store data that is accessible by every function in the pipeline.

Further improvements
1- The binary data extraction method used is very limited to near-perfect light condition. This is the main reason of the system failure when testing using the challenge video. 
2- Sanity checks could help improve the tracking quality.
