##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/dataset.png
[image2]: ./output_images/HOG_Features.jpg
[image3]: ./output_images/Boxes_1.jpg
[image4]: ./output_images/Search_Area_1.jpg
[image5]: ./output_images/Search_Area_2.png
[image6]: ./output_images/Search_Area_3.png
[image7]: ./output_images/Search_Area_4.png
[image8]: ./output_images/Combined_Search_Area.png
[image9]: ./output_images/Heatmap.png
[image10]: ./output_images/Heatmap_plus_Threshold.png
[image11]: ./output_images/Scipy_labels.png
[image12]: ./output_images/Boxes_on_heatmap.png
[image13]: ./output_images/Final_Result.png
[video1]: ./project_video_out_2017_10_28_18_8_15.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the fifth code cell of the IPython notebook `Vehicle_Detection.ipynb` 

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YUV` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(8, 8)`:


![alt text][image2]

The method `extract_features` takes an image and returns a flattened array of HOG features for each image in the list.

The table below documents ten different parameter settings I explored.

| Configuration Label | Colorspace | Orientations | Pixels Per Cell | Cells Per Block | HOG Channel | Extract Time |
| :-----------------: | :--------: | :----------: | :-------------: | :-------------: | :---------: | ------------:|
| 1                   | YUV        | 9            | 8               | 1               | 0           | 44.04        |
| 2                   | YUV        | 9            | 8               | 3               | 0           | 37.74        |
| 3                   | YUV        | 6            | 8               | 2               | 0           | 37.12        |
| 4                   | YUV        | 12           | 8               | 2               | 0           | 40.11        |
| 5                   | YUV        | 11           | 8               | 2               | 0           | 38.01        |
| 6                   | YUV        | 11           | 16              | 2               | 0           | 30.21        |
| 7                   | YUV        | 11           | 12              | 2               | 0           | 30.33        |
| 8                   | YUV        | 11           | 4               | 2               | 0           | 69.08        |
| 9                   | YUV        | 11           | 16              | 2               | ALL         | 55.20        |
| 10                  | YUV        | 7            | 16              | 2               | ALL         | 53.18        |

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and the final choice was based on:
- The performance of the SVM classifier produced using them
- Total training time

The table below describes performance and their corresponding configurations.

| Configuration (above) | Classifier | Accuracy | Train Time |
| :-------------------: | :--------: | -------: | ---------: |
| 1                     | Linear SVC | 94.76    | 5.11       |
| 2                     | Linear SVC | 96.11    | 6.71       |
| 3                     | Linear SVC | 95.81    | 3.79       |
| 4                     | Linear SVC | 95.95    | 4.84       |
| 5                     | Linear SVC | 96.59    | 5.46       |
| 6                     | Linear SVC | 95.16    | 0.52       |
| 7                     | Linear SVC | 94.85    | 1.27       |
| 8                     | Linear SVC | 95.92    | 21.39      |
| 9                     | Linear SVC | 98.17    | 1.14       |
| 10                    | Linear SVC | 97.61    | 1.42       |

The final parameters chosen were those labeled "configuration 9" in the table above: YUV colorspace, 11 orientations, 16 pixels per cell, 2 cells per block, and `ALL` channels of the colorspace.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM with the default classifier parameters and using HOG features alone and was able to achieve a test accuracy of 98.17%.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The method `find_cars` combines HOG feature extraction with a sliding window search, but rather than performing feature extraction on each window individually, the HOG features are extracted for the entire image and then these full-image features are subsampled according to the size of the window and then fed to the classifier. The method performs the classifier prediction on the HOG features for each window region and returns a list of rectangle objects corresponding to the windows that generated a positive ("car") prediction.

The image below shows the first attempt at using `find_cars` on one of the test images, using a single window size:

![alt text][image3]

I also explored several configurations of window size. The results are as below:

![alt text][image4]

![alt text][image5]

![alt text][image6]

![alt text][image7]

The below image is the result of combining above search windows:

![alt text][image8]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image13]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out_2017_10_28_18_8_15.mp4)

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image9]

![alt text][image10]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image11]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image12]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further. 

Integrating detections from previous frames mitigates the effect of the misclassifications, but it also introduces another problem: vehicles that significantly change position from one frame to the next will tend to escape being labeled.

The pipeline will fail in cases where vehicles don't resemble those in the training dataset, but lighting and environmental conditions might also play a role (e.g. a white car against a white background)

