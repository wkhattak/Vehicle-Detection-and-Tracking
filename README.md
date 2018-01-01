# Project: Advanced Lane Finding
## Overview   
   
This project is about writing a software pipeline to detect cars in a video using an [SVM classifier](https://en.wikipedia.org/wiki/Support_vector_machine) trained on [HOG](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients) features.

## How Does It Work?
The logic comprises an image processing pipeline with the following steps:

1. Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
2. Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
5. Run the image processing pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
6. Estimate a bounding box for vehicles detected and draw the bounding box on the image.

For a detailed description of this project, please view the [project writeup](./writeup.md).

## Directory Structure
* **test_images:** Directory containing sample images for testing
* **output_images:** Directory containing images used in the writeup
* **saved_model:** Directory containing pickled SVM classifer and the training parameters
* **project_video.mp4:** Video clip for testing the image processing pipeline
* **project_video_output.mp4:** Processed video clip via the image processing pipeline
* **Project.html:** Html output of the Jupyter notebook
* **Project.ipynb:** Jupyter notebook containing the Python source code
* **README.md:** Project readme file
* **writeup.md:** project writeup file containing detailed information about the inner workings of this project

## Requirements
* Python-3.5.2
* OpenCV-3.3.0
* numpy
* sklearn
* skimage
* scipy
* matplotlib
* moviepy
* Jupyter Notebook


## Usage/Examples

Load the already trained model by running the code cells: *Load Trained Model & Parameters*, *Car Detection*, *Car Detection*, *Heatmap Class* and *Main Pipeline*. 

**For individual images:**

```python
# Heatmap object
heatmap_obj = HeatMap()
# Y-axis dimensions & scales for sliding window search
ydims_scales = [(400,536,1),(400,578,1.3),(400,604,1.5),(400,630,1.7),(400,672,2),(400,680,2.5),(400,664,3),(400,708,3.5)]
# Model's min. prediction confidence below which the detections are discarded
min_prediction_confidence = 0.25
# Min. bounding box dimensions below which the bounding box is not drawn on screen 
min_bbox_w_h = (22,48)
# Debug flag
debug = True
# Detailed debug flag
debug_full = False
# Heatmap overlap threshold (if less than or equal to then the pixels are turned off in the heatmap)
duplicate_threshold = 1
# X-axis value below which the bounding box is not drawn on the screen
x_threshold = 640
# Log file location
log_file = open('./log.txt', 'w')               
# Counter used for naming images (used in visualization & logging)
frame_counter = 0
# Whether to use averaged heatmap of previous 15 frames (if false then only the current frame is considered)
use_avg_heatmap = False

img_path = './test_images/test1.jpg'
img = mpimg.imread(img_path)
frame_name = img_path.split('/')[-1]
out_img = process_frame(img)
fig, ax = plt.subplots(figsize=(15,10))
ax.imshow(out_img)
ax.set_title('Detected Cars', fontsize=20)
plt.show()
log_file.close()
```

**For video clip:**

```python
# Heatmap object
heatmap_obj = HeatMap()
# Y-axis dimensions & scales for sliding window search
ydims_scales = [(400,536,1),(400,578,1.3),(400,604,1.5),(400,630,1.7),(400,672,2),(400,680,2.5),(400,664,3),(400,708,3.5)]
# Model's min. prediction confidence below which the detections are discarded
min_prediction_confidence = 0.25
# Min. bounding box dimensions below which the bounding box is not drawn on screen 
min_bbox_w_h = (22,48)
# Debug flag
debug = False
# Detailed debug flag
debug_full = False
# Heatmap overlap threshold (if less than or equal to then the pixels are turned off in the heatmap)
duplicate_threshold = 1
# X-axis value below which the bounding box is not drawn on the screen
x_threshold = 640
# Log file location
log_file = open('./log.txt', 'w')               
# Counter used for naming images (used in visualization & logging)
frame_counter = 0
# Whether to use averaged heatmap of previous 15 frames (if false then only the current frame is considered)
use_avg_heatmap = True
# Used for logging
frame_name= None

video_output_raw = './project_video_output.mp4'
clip1_raw = VideoFileClip('./project_video.mp4')
video_clip_raw = clip1_raw.fl_image(process_frame) 
%time video_clip_raw.write_videofile(video_output_raw, audio=False)
log_file.close()
```

## Troubleshooting

**ffmpeg**

NOTE: If you don't have ffmpeg installed on your computer you'll have to install it for moviepy to work. If this is the case you'll be prompted by an error in the notebook. You can easily install ffmpeg by running the following in a code cell in the notebook.

```python
import imageio
imageio.plugins.ffmpeg.download()
```

Once it's installed, moviepy should work.

## License
The content of this project is licensed under the [Creative Commons Attribution 3.0 license](https://creativecommons.org/licenses/by/3.0/us/deed.en_US).