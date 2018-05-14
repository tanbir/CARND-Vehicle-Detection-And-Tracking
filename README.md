# Udacity Self-Driving Car Engineer Nanodegree Program
## Vehicle Detection Project

The goals / steps of this project are the following:

---

* Performing a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier
    * Optionally, applying a color transform and appending binned color features, as well as histograms of color, to the HOG feature vector. 
    * Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implementing a sliding-window technique and using the trained classifier to search for vehicles in images.
* Running the pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and creating a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimating a bounding box for vehicles detected.

---

* All of the code for the project is contained in the Jupyter notebook `vehicle_detection_pipeline.ipynb` 
* The class that implements the vehicle detection algorithms is `VehicleDetect`
---

[//]: # (Image References)
[image1]: ./output_images/random_images.png
[image2]: ./output_images/hog_features.png
[image3]: ./output_images/sliding_window_1.png
[image4]: ./output_images/sliding_window_2.png
[image5]: ./output_images/pipeline_on_test.png
[video1]: ./test_video_out.mp4
[video2]: ./project_video_out.mp4

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

Following are the steps I used:

* Loading (using the method `load_data`) all of the vehicle and non-vehicle image paths from the provided dataset. The figure below shows an example: 

![alt text][image1]

* The code for extracting HOG features from an image is defined by the method `get_hog_features`. The figure below shows a comparison of car images and associated histograms of oriented gradients, as well as the same for non-car images.

![alt text][image2]

* The method `extract_features` accepts a list of image paths and HOG parameters (as well as one of a variety of destination color spaces, to which the input image is converted), and produces a flattened array of HOG features for each image in the list.

#### 2. Explain how you settled on your final choice of HOG parameters.

Follwoing is my exploration of the parameter space:

| Index | Colorspace |  HOG Channel |Orientations | Pixels Per Cell | Cells Per Block | Extract Time | Vector size | Training time | Accuracy |
| :-----------------: | :--------: | :----------: | :-------------: | :-------------: | :---------: | :------------:|:------------:|:------------:|:------------:|
| 1 | HLS | 0 | 9 | 8 | 2 | 24.61 | 1764 | 5.02 | 0.9226 |
| 2 | HLS | 0 | 9 | 16 | 2 | 35.1 | 324 | 0.51 | 0.9119 |
| 3 | HLS | 1 | 9 | 8 | 2 | 40.38 | 1764 | 3.71 | 0.9679 |
| 4 | HLS | 1 | 9 | 16 | 2 | 30.29 | 324 | 0.47 | 0.9499 |
| 5 | HLS | 2 | 9 | 8 | 2 | 37.13 | 1764 | 5.99 | 0.9088 |
| 6 | HLS | 2 | 9 | 16 | 2 | 29.44 | 324 | 0.49 | 0.8801 |
| 7 | HLS | ALL | 9 | 8 | 2 | 89.63 | 5292 | 7.77 | 0.9854 |
| 8 | HLS | ALL | 9 | 16 | 2 | 78.67 | 972 | 1.28 | 0.9766 |
| 9 | HLS | 0 | 10 | 8 | 2 | 26.55 | 1960 | 3.89 | 0.9398 |
| 10 | HLS | 0 | 10 | 16 | 2 | 32.94 | 360 | 0.41 | 0.9189 |
| 11 | HLS | 1 | 10 | 8 | 2 | 37.07 | 1960 | 3.54 | 0.955 |
| 12 | HLS | 1 | 10 | 16 | 2 | 28.01 | 360 | 0.51 | 0.9428 |
| 13 | HLS | 2 | 10 | 8 | 2 | 33.66 | 1960 | 5.22 | 0.9116 |
| 14 | HLS | 2 | 10 | 16 | 2 | 28.56 | 360 | 0.47 | 0.8801 |
| 15 | HLS | ALL | 10 | 8 | 2 | 85.3 | 5880 | 7.23 | 0.9837 |
| 16 | HLS | ALL | 10 | 16 | 2 | 87.69 | 1080 | 0.85 | 0.978 |
| 17 | HLS | 0 | 11 | 8 | 2 | 24.67 | 2156 | 4.62 | 0.9274 |
| 18 | HLS | 0 | 11 | 16 | 2 | 18.44 | 396 | 0.59 | 0.9133 |
| 19 | HLS | 1 | 11 | 8 | 2 | 24.4 | 2156 | 3.9 | 0.964 |
| 20 | HLS | 1 | 11 | 16 | 2 | 18.09 | 396 | 0.45 | 0.9485 |
| 21 | HLS | 2 | 11 | 8 | 2 | 24.86 | 2156 | 5.46 | 0.9093 |
| 22 | HLS | 2 | 11 | 16 | 2 | 17.8 | 396 | 0.65 | 0.8789 |
| 23 | HLS | ALL | 11 | 8 | 2 | 63.65 | 6468 | 8.26 | 0.9783 |
| 24 | HLS | ALL | 11 | 16 | 2 | 43.03 | 1188 | 1.41 | 0.9755 |
| 25 | HSV | 0 | 9 | 8 | 2 | 24.67 | 1764 | 4.92 | 0.9358 |
| 26 | HSV | 0 | 9 | 16 | 2 | 29.22 | 324 | 0.53 | 0.9144 |
| 27 | HSV | 1 | 9 | 8 | 2 | 35.24 | 1764 | 4.82 | 0.9155 |
| 28 | HSV | 1 | 9 | 16 | 2 | 27.14 | 324 | 0.46 | 0.8986 |
| 29 | HSV | 2 | 9 | 8 | 2 | 31.98 | 1764 | 3.46 | 0.9597 |
| 30 | HSV | 2 | 9 | 16 | 2 | 25.81 | 324 | 0.46 | 0.9403 |
| 31 | HSV | ALL | 9 | 8 | 2 | 73.91 | 5292 | 7.34 | 0.9851 |
| 32 | HSV | ALL | 9 | 16 | 2 | 58.42 | 972 | 0.93 | 0.9794 |
| 33 | HSV | 0 | 10 | 8 | 2 | 23.96 | 1960 | 3.57 | 0.9367 |
| 34 | HSV | 0 | 10 | 16 | 2 | 30.36 | 360 | 0.39 | 0.931 |
| 35 | HSV | 1 | 10 | 8 | 2 | 35.57 | 1960 | 4.65 | 0.9195 |
| 36 | HSV | 1 | 10 | 16 | 2 | 28.91 | 360 | 0.5 | 0.8956 |
| 37 | HSV | 2 | 10 | 8 | 2 | 34.66 | 1960 | 3.63 | 0.962 |
| 38 | HSV | 2 | 10 | 16 | 2 | 25.04 | 360 | 0.49 | 0.9457 |
| 39 | HSV | ALL | 10 | 8 | 2 | 81.57 | 5880 | 7.33 | 0.9831 |
| 40 | HSV | ALL | 10 | 16 | 2 | 67.59 | 1080 | 0.84 | 0.982 |
| 41 | HSV | 0 | 11 | 8 | 2 | 25.22 | 2156 | 5.07 | 0.9412 |
| 42 | HSV | 0 | 11 | 16 | 2 | 18.73 | 396 | 0.59 | 0.9203 |
| 43 | HSV | 1 | 11 | 8 | 2 | 25.1 | 2156 | 4.71 | 0.911 |
| 44 | HSV | 1 | 11 | 16 | 2 | 19.2 | 396 | 0.64 | 0.8947 |
| 45 | HSV | 2 | 11 | 8 | 2 | 24.42 | 2156 | 3.78 | 0.962 |
| 46 | HSV | 2 | 11 | 16 | 2 | 18.34 | 396 | 0.45 | 0.9513 |
| 47 | HSV | ALL | 11 | 8 | 2 | 63.63 | 6468 | 7.53 | 0.984 |
| 48 | HSV | ALL | 11 | 16 | 2 | 43.43 | 1188 | 1.27 | 0.9786 |
| 49 | LUV | 0 | 9 | 8 | 2 | 24.69 | 1764 | 3.75 | 0.9617 |
| 50 | LUV | 0 | 9 | 16 | 2 | 30.72 | 324 | 0.37 | 0.942 |
| 51 | LUV | 1 | 9 | 8 | 2 | 36.66 | 1764 | 3.88 | 0.9367 |
| 52 | LUV | 1 | 9 | 16 | 2 | 24.75 | 324 | 0.44 | 0.9279 |
| 53 | LUV | 2 | 9 | 8 | 2 | 30.49 | 1764 | 4.61 | 0.9167 |
| 54 | LUV | 2 | 9 | 16 | 2 | 26.22 | 324 | 0.46 | 0.9127 |
| 55 | LUV | ALL | 9 | 8 | 2 | 82.5 | 5292 | 9.57 | 0.9764 |
| 56 | LUV | ALL | 9 | 16 | 2 | 70.35 | 972 | 1.15 | 0.9772 |
| 57 | LUV | 0 | 10 | 8 | 2 | 26.95 | 1960 | 3.71 | 0.9572 |
| 58 | LUV | 0 | 10 | 16 | 2 | 30.41 | 360 | 0.44 | 0.9451 |
| 59 | LUV | 1 | 10 | 8 | 2 | 36.09 | 1960 | 3.84 | 0.9341 |
| 60 | LUV | 1 | 10 | 16 | 2 | 26.1 | 360 | 0.42 | 0.9181 |
| 61 | LUV | 2 | 10 | 8 | 2 | 30.21 | 1960 | 4.36 | 0.9268 |
| 62 | LUV | 2 | 10 | 16 | 2 | 26.18 | 360 | 0.42 | 0.9009 |
| 63 | LUV | ALL | 10 | 8 | 2 | 80.95 | 5880 | 9.67 | 0.9764 |
| 64 | LUV | ALL | 10 | 16 | 2 | 73.48 | 1080 | 1.01 | 0.9727 |
| 65 | LUV | 0 | 11 | 8 | 2 | 24.84 | 2156 | 3.87 | 0.9595 |
| 66 | LUV | 0 | 11 | 16 | 2 | 18.32 | 396 | 0.57 | 0.951 |
| 67 | LUV | 1 | 11 | 8 | 2 | 25.57 | 2156 | 3.88 | 0.9465 |
| 68 | LUV | 1 | 11 | 16 | 2 | 18.98 | 396 | 0.5 | 0.9231 |
| 69 | LUV | 2 | 11 | 8 | 2 | 24.57 | 2156 | 4.47 | 0.9245 |
| 70 | LUV | 2 | 11 | 16 | 2 | 19.02 | 396 | 0.6 | 0.9048 |
| 71 | LUV | ALL | 11 | 8 | 2 | 62.58 | 6468 | 7.91 | 0.9806 |
| 72 | LUV | ALL | 11 | 16 | 2 | 44.07 | 1188 | 1.32 | 0.9752 |
| 73 | RGB | 0 | 9 | 8 | 2 | 23.43 | 1764 | 3.81 | 0.9595 |
| 74 | RGB | 0 | 9 | 16 | 2 | 18.26 | 324 | 0.53 | 0.9412 |
| 75 | RGB | 1 | 9 | 8 | 2 | 24.3 | 1764 | 3.29 | 0.962 |
| 76 | RGB | 1 | 9 | 16 | 2 | 20.86 | 324 | 0.48 | 0.9496 |
| 77 | RGB | 2 | 9 | 8 | 2 | 26.78 | 1764 | 3.36 | 0.9561 |
| 78 | RGB | 2 | 9 | 16 | 2 | 21.07 | 324 | 0.41 | 0.951 |
| 79 | RGB | ALL | 9 | 8 | 2 | 70.33 | 5292 | 20.16 | 0.9733 |
| 80 | RGB | ALL | 9 | 16 | 2 | 53.87 | 972 | 2.19 | 0.9699 |
| 81 | RGB | 0 | 10 | 8 | 2 | 23.9 | 1960 | 3.42 | 0.9513 |
| 82 | RGB | 0 | 10 | 16 | 2 | 23.36 | 360 | 0.59 | 0.9409 |
| 83 | RGB | 1 | 10 | 8 | 2 | 28.49 | 1960 | 3.38 | 0.9657 |
| 84 | RGB | 1 | 10 | 16 | 2 | 22.84 | 360 | 0.51 | 0.9516 |
| 85 | RGB | 2 | 10 | 8 | 2 | 28.61 | 1960 | 3.3 | 0.9578 |
| 86 | RGB | 2 | 10 | 16 | 2 | 24.08 | 360 | 0.47 | 0.9462 |
| 87 | RGB | ALL | 10 | 8 | 2 | 78.61 | 5880 | 14.86 | 0.9654 |
| 88 | RGB | ALL | 10 | 16 | 2 | 72.31 | 1080 | 2.26 | 0.9628 |
| 89 | RGB | 0 | 11 | 8 | 2 | 23.31 | 2156 | 3.94 | 0.9611 |
| 90 | RGB | 0 | 11 | 16 | 2 | 17.25 | 396 | 0.62 | 0.9403 |
| 91 | RGB | 1 | 11 | 8 | 2 | 23.21 | 2156 | 3.43 | 0.9642 |
| 92 | RGB | 1 | 11 | 16 | 2 | 17.03 | 396 | 0.54 | 0.9516 |
| 93 | RGB | 2 | 11 | 8 | 2 | 23.33 | 2156 | 3.53 | 0.9617 |
| 94 | RGB | 2 | 11 | 16 | 2 | 17.16 | 396 | 0.58 | 0.9555 |
| 95 | RGB | ALL | 11 | 8 | 2 | 60.35 | 6468 | 13.54 | 0.9702 |
| 96 | RGB | ALL | 11 | 16 | 2 | 40.98 | 1188 | 2.54 | 0.9654 |
| 97 | YCrCb | 0 | 9 | 8 | 2 | 22.94 | 1764 | 3.58 | 0.9693 |
| 98 | YCrCb | 0 | 9 | 16 | 2 | 22.96 | 324 | 0.38 | 0.9465 |
| 99 | YCrCb | 1 | 9 | 8 | 2 | 28.62 | 1764 | 3.61 | 0.9476 |
| 100 | YCrCb | 1 | 9 | 16 | 2 | 23.64 | 324 | 0.39 | 0.9358 |
| 101 | YCrCb | 2 | 9 | 8 | 2 | 30.16 | 1764 | 4.09 | 0.933 |
| 102 | YCrCb | 2 | 9 | 16 | 2 | 24.22 | 324 | 0.43 | 0.9248 |
| 103 | YCrCb | ALL | 9 | 8 | 2 | 77.76 | 5292 | 6.49 | 0.9786 |
| 104 | YCrCb | ALL | 9 | 16 | 2 | 62.76 | 972 | 0.91 | 0.9797 |
| 105 | YCrCb | 0 | 10 | 8 | 2 | 25.16 | 1960 | 3.33 | 0.9597 |
| 106 | YCrCb | 0 | 10 | 16 | 2 | 25.06 | 360 | 0.47 | 0.9428 |
| 107 | YCrCb | 1 | 10 | 8 | 2 | 30.9 | 1960 | 3.09 | 0.9462 |
| 108 | YCrCb | 1 | 10 | 16 | 2 | 24.81 | 360 | 0.35 | 0.9459 |
| 109 | YCrCb | 2 | 10 | 8 | 2 | 29.77 | 1960 | 3.75 | 0.944 |
| 110 | YCrCb | 2 | 10 | 16 | 2 | 25.17 | 360 | 0.4 | 0.9268 |
| 111 | YCrCb | ALL | 10 | 8 | 2 | 74.08 | 5880 | 6.22 | 0.984 |
| 112 | YCrCb | ALL | 10 | 16 | 2 | 69.38 | 1080 | 0.8 | 0.9825 |
| 113 | YCrCb | 0 | 11 | 8 | 2 | 23.56 | 2156 | 3.69 | 0.9614 |
| 114 | YCrCb | 0 | 11 | 16 | 2 | 17.33 | 396 | 0.54 | 0.9561 |
| 115 | YCrCb | 1 | 11 | 8 | 2 | 24.02 | 2156 | 3.46 | 0.9476 |
| 116 | YCrCb | 1 | 11 | 16 | 2 | 18.28 | 396 | 0.47 | 0.9344 |
| 117 | YCrCb | 2 | 11 | 8 | 2 | 24.44 | 2156 | 4.28 | 0.9358 |
| 118 | YCrCb | 2 | 11 | 16 | 2 | 17.57 | 396 | 0.49 | 0.9288 |
| 119 | YCrCb | ALL | 11 | 8 | 2 | 61.37 | 6468 | 6.6 | 0.987 |
| 120 | YCrCb | ALL | 11 | 16 | 2 | 42.22 | 1188 | 1.02 | 0.98 |
| 121 | YUV | 0 | 9 | 8 | 2 | 23.06 | 1764 | 3.65 | 0.9651 |
| 122 | YUV | 0 | 9 | 16 | 2 | 24.22 | 324 | 0.48 | 0.9499 |
| 123 | YUV | 1 | 9 | 8 | 2 | 31.16 | 1764 | 3.84 | 0.9313 |
| 124 | YUV | 1 | 9 | 16 | 2 | 23.84 | 324 | 0.41 | 0.924 |
| 125 | YUV | 2 | 9 | 8 | 2 | 30.36 | 1764 | 3.65 | 0.9496 |
| 126 | YUV | 2 | 9 | 16 | 2 | 24.29 | 324 | 0.39 | 0.9299 |
| 127 | YUV | ALL | 9 | 8 | 2 | 76.82 | 5292 | 7.23 | 0.9806 |
| 128 | YUV | ALL | 9 | 16 | 2 | 62.6 | 972 | 0.9 | 0.9817 |
| 129 | YUV | 0 | 10 | 8 | 2 | 25.89 | 1960 | 3.73 | 0.9609 |
| 130 | YUV | 0 | 10 | 16 | 2 | 24.83 | 360 | 0.51 | 0.9448 |
| 131 | YUV | 1 | 10 | 8 | 2 | 29.94 | 1960 | 3.57 | 0.9369 |
| 132 | YUV | 1 | 10 | 16 | 2 | 24.79 | 360 | 0.39 | 0.9274 |
| 133 | YUV | 2 | 10 | 8 | 2 | 29.83 | 1960 | 3.29 | 0.9538 |
| 134 | YUV | 2 | 10 | 16 | 2 | 24.63 | 360 | 0.36 | 0.9437 |
| 135 | YUV | ALL | 10 | 8 | 2 | 83.18 | 5880 | 7.99 | 0.9834 |
| 136 | YUV | ALL | 10 | 16 | 2 | 81.36 | 1080 | 0.83 | 0.9825 |
| 137 | YUV | 0 | 11 | 8 | 2 | 23.35 | 2156 | 3.58 | 0.9642 |
| 138 | YUV | 0 | 11 | 16 | 2 | 17.5 | 396 | 0.56 | 0.949 |
| 139 | YUV | 1 | 11 | 8 | 2 | 24.38 | 2156 | 4.1 | 0.9338 |
| 140 | YUV | 1 | 11 | 16 | 2 | 17.5 | 396 | 0.47 | 0.9299 |
| 141 | YUV | 2 | 11 | 8 | 2 | 23.94 | 2156 | 3.54 | 0.9437 |
| 142 | YUV | 2 | 11 | 16 | 2 | 17.49 | 396 | 0.45 | 0.9459 |
| 143 | YUV | ALL | 11 | 8 | 2 | 60.23 | 6468 | 6.33 | 0.982 |
| 144 | YUV | ALL | 11 | 16 | 2 | 41.84 | 1188 | 0.97 | 0.9803 |

We consider the following parameters to optimize time for feature extraction and training as well as test accuracy:

* Color space = `YUV`
* Orientation = 11
* Pixels per cell = 16
* Cells per block = 2
* Hog channels = `ALL`


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In the method `train()`, I trained a linear SVM with the default classifier parameters and using HOG features alone (I did not use spatial intensity or channel intensity histogram features) and was able to achieve a test accuracy of 98.56%. 


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

* I have explored the following configurations of the window sizes and positions using the method `potential_search_areas`

![alt text][image3]

* The method `update_heatmap` increments the pixel value (referred to as "heat") of an all-black image the size of the original image at the location of each detection rectangle. Areas encompassed by more overlapping rectangles are assigned higher levels of heat.

* A threshold by the method `apply_threshold` is applied to the heatmap (in this example, with a value of 1), setting all pixels that don't exceed the threshold to zero. 

* The `scipy.ndimage.measurements.label()` function collects spatially contiguous areas of the heatmap and assigns each a label

* Finally, the bounding boxes are drawn by `draw_bounding_bboxes(...)`

![alt text][image4]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The results of passing all of the project test images through the pipeline are displayed in the images below:

![alt text][image5]


---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result][video2]

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

* The code for processing frames of video is contained in the method `process_frames` 
* For single images, the procedure is as described above
* For videos frames, caches the previous 15 frames so that detections from the past 15 frames can be combined and added to the heatmap and thresholding

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* Problems faced:
    * Obtaining a high test accuracy (solved using extensive parameter search)
    * Flicker of video bounding boxes (solved using cached frames)

* Possible issues:
    * May fail on various lighting conditions, new kind of vehicles, as well as on variation of sizes and distances of the vehicles

* Alternatives:
    * Using a CNN to have a deep-learning solution to the computer vision problem