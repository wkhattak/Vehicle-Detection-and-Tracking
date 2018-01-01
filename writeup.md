# Vehicle Detection and Tracking Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

## [Rubric Points](https://review.udacity.com/#!/rubrics/513/view) 
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained across multiple cells, namely *Load Training Images*, *Visualize Training Images*, *Parameters*, *Feature Extraction* and *Visualize HOG Features*. The primary function used for creating the HOG features is `get_hog_features()` that used the parameters defined in the *Parameters* code cell, whereas the `extract_features()` function is used to call `get_hog_features()` as well as `bin_spatial()` & `color_hist()` functions to extract *binned color* and *color histogram* features respectively in preparation for model training.

I started by reading in all the *vehicle* and *non-vehicle* images.  Here is an example of 10 of each of the *vehicle* and *non-vehicle* images along with information about the total number of images in each category:

![training data](/output_images/training_data_exploration.jpg)

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example of *car* & *non-car* HOG features using the *YUV* color space and HOG parameters of `orientations=11`, `pixels_per_cell=(12, 12)` and `cells_per_block=(2, 2)`:


![hog](/output_images/hog_visualization.jpg)

#### 2. Explain how you settled on your final choice of HOG parameters.

Having familiarized myself with *HOG feature* extraction, as well as *binned color* and *color histogram* feature extraction, I then looked in to using these features for training a model. The idea was to use different combinations of parameters to find the optimum combination both in terms of model accuracy as well as the execution time.

The following table summarizes my findings:

|Color|Spatial Bin.|Color Hist.|HOG Channels|HOG Orient.|HOG Pix./Cell|HOG Cells/Block|Feature Extrac. Time|Train. Time|Accuracy|
|:------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|HLS|Yes|Yes|All|9|8|2|401.06|15.88|0.9913|
|HLS|No|No|All|9|8|2|372.04|64.46|0.9783|
|HSV|No|No|All|9|8|2|377.77|16.67|0.9783|
|LUV|No|No|All|9|8|2|377.46|63.22|0.9772|
|YCrCb|No|No|All|9|8|2|377.22|56.81|0.9823|
|RGB|No|No|All|9|8|2|379.24|75.35|0.9679|
|YUV|No|No|All|9|8|2|379.22|55.97|0.9764|
|YUV|No|No|All|11|16|2|271.28|9.09|0.9733|
|YUV|No|No|All|11|12|2|285.50|18.55|0.9764|

*All times in seconds*

Based on the accuracy obtained by using *HOG*, *binned color* and *color histogram* features, at first I chose this combination pf parameters even though the feature extraction time was quite high. However, tests revealed poor detection performance that is believed to be a consequence of overfitting. Next, I started experimenting with using only HOG features for the model. All combinations more or less 
resulted in the same model accuracy, however, further tests revealed *YUV* color space to be performing better than other color spaces, especially under different lighting situations e.g. light colored tarmac. Based on this I chose *YUV* color space with initial HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`.

Then, I looked at optimizing the model training time as using the above parameter combination was too time consuming. Specifically, I wanted to bring down the *feature extraction time* and *training time*. I first tried a combination of `orientations=11`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)` that significantly brought down the *feature extraction time* and *training time* without much loss in *model accuracy*. But using this combination bore very poor detection performance. After further exploration, a combination of `orientations=11`, `pixels_per_cell=(12, 12)` and `cells_per_block=(2, 2)` was chosen as it provided the best execution speed and vehicle detection performance.

Below table shows execution times using different parameter combinations for the *YUV* color space:

|Color|HOG Channels|HOG Orient.|HOG Pix./Cell|HOG Cells/Block|Video Frames|Processing Time|
|:------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|YUV|All|12|8|2|100|11|
|YUV|All|11|16|2|100|4.04|
|YUV|All|11|12|2|100|6.04|

*All times in minutes*

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In the *Model Training* code cell, I trained the classifier by first extracting the HOG features and then feeding them to a linear SVM using the following parameter combination:

```
python

color_space = 'YUV'
orient = 11  
pix_per_cell = 12 
cell_per_block = 2 
hog_channel = 'ALL' 
spatial_feat = False 
hist_feat = False 
hog_feat = True 
```

Further, I normalized the HOG features using the `sklearn.preprocessing.StandardScaler()` class so that the features have 0 mean and unit variance. Then I trained the model on 80% of the training data while keeping 20% for testing.

Following image summarizes the model training process:

![model training](/output_images/model_training.jpg)


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I implemented the sliding window search in the function `find_cars()` (code cell *Car Detection*). This function takes in an array of tuples `ydims_scales`. The first 2 elements are the *start* and *end* positions on y-axis of the image for specifying only that area in the image where cars might appear and the last element of the tuple specifies the window scale. Searching in a sub-area of the image greatly reduces the execution time & further reduces false positives at smaller scales. Then the image is resized based on the scale and the number of blocks for both x & y direction are then calculated. Also, the number of blocks that would fit into the search window are calculated, based on the default window size of `64 x 64` pixels. The scale increases or decreases the window size e.g. with a scale of `1.5`, the resulting window size is `96x96` pixels. For controlling the overlap behavior, `cells_per_step` setting is used. With `cells_per_step = 2` and a scale of `1`, the overlap is `62.50%` in both x & y directions (for other scales, see the table below). A setting of `cells_per_step = 1` is used for scales above `2` to increase the possibility of detecting more cars at those scales, as the window size is larger than normal and with using `cells_per_step = 2`, either there is no overlap or areas are missed at the right side of the image. 

After extensive experimenting, I settled for these scales: `1`,`1.3`,`1.5`,`1.7`,`2.0`,`2.5`,`3.0` & `3.5`. Each of these scales are restricted in vertical direction as smaller scales tend to not only result in more false positives but also increase execution time if searched for the entire area where cars might appear. For each scale, the HOG features are only extracted once and then sub-sampled for each sliding search window, which decreases the execution time.

The following series of images show the sliding windows and their overlap at the aforementioned scales:


![scale 1](/output_images/scale_1.jpg)

![scale 1.3](/output_images/scale_1_3.jpg)

![scale 1.5](/output_images/scale_1_5.jpg)

![scale 1.7](/output_images/scale_1_7.jpg)

![scale 2](/output_images/scale_2.jpg)

![scale 2.5](/output_images/scale_2.5.jpg)

![scale 3](/output_images/scale_3.jpg)

![scale 3.5](/output_images/scale_3_5.jpg)

**Sliding window Scales**

|Scale|Size (wxh)|Overlap % (x & y)|Window Step (pixels)|No. of Windows (vertically)|Start (y-axis)|End (y-axis)|
|:------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|1.0|64x64|62.50|24|3|400|536|
|1.3|83x83|62.74|31|3|400|578|
|1.5|96x96|62.50|36|3|400|604|
|1.7|108x108|63.23|40|3|400|630|
|2.0|128x128|62.50|48|3|400|672|
|2.5|160x160|81.25|30|4|400|680|
|3.0|192x192|81.25|36|2|400|664|
|3.5|224x224|81.25|42|2|400|708|

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

To optimize the performance of the classifier, I took the following actions:

1. Changed the HOG parameters from `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` to `orientations=11`, `pixels_per_cell=(12, 12)` and `cells_per_block=(2, 2)`.

2. Only used HOG feature, excluding *spatial binning* and *color histogram* features.

To reduce the execution time of the pipeline, I took the following actions:

1. Vertically restricted window search to only that area where vehicles normally appear on the road.

2. Extracted HOG features only once for the entire image and then retrieved the HOG features for the current window area (sub-sampling).

The pipeline is implemented via the `process_frame()` function (code cell *Main Pipeline*). The following images demonstrate the application of the pipeline on test images:

![pipeline test](/output_images/pipeline_test_images.jpg)

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The code for the implementation of filtering false positives is spread across the cells *Car Detection* and *Main Pipeline* cells while the code for combining overlapping bounding boxes can be found in the *Car Detection* code cell.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap, `add_heat()` function, and then thresholded the heatmap, `apply_threshold()` function, to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` function to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes, `draw_labeled_bboxes()` function, to cover the area of each blob detected.  

The first stage of filtering false positives is implemented in the `find_cars()` function. The classifier's [decision_function()](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC.decision_function()) provides the prediction confidence score. If this score is less than the `min_prediction_confidence` value or if the top-left x position of the detected bounding box is less than the `x_threshold`, the detection is ignored.

The second stage of the false positive identification is implemented in the `draw_labeled_bboxes()` function. Here I am checking if the bounding box dimensions are less than the `min_bbox_w_h` value, then it is discarded.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

