# **Traffic Sign Recognition** 

## Writeup

### Presented By Medhat HUSSAIN

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./00_out/histogram.png "Visualization"
[image2]: ./00_out/normalized.png "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./01_test_images/small_120_kmh_limit.png "Traffic Sign 1"
[image5]: ./01_test_images/small_no_vehicles.png "Traffic Sign 2"
[image6]: ./01_test_images/small_priority_road.png "Traffic Sign 3"
[image7]: ./01_test_images/small_road_works.png "Traffic Sign 4"
[image8]: ./01_test_images/small_vehicles_over_3.5_tonnes_prohibited.png "Traffic Sign 5"
[image9]: ./00_out/validation_accuracy.jpg "Grayscaling"
[image10]: ./00_out/loss.jpg "Grayscaling"
[image11]: ./00_out/noise2.png "Grayscaling"
[image12]: ./00_out/noise5.png "Grayscaling"
[image13]: ./00_out/blur.png "Grayscaling"
[image14]: ./00_out/X_EqualizeHist.png "Grayscaling"
[image15]: ./00_out/X_EqualizeHist4.png "Grayscaling"
[image16]: ./00_out/final.png "Grayscaling"
[image1616]: ./00_out/final1.png "Grayscaling"
[image17]: ./00_out/5_photos.png "Grayscaling"
[image18]: ./00_out/5_photos.png "Grayscaling"
[image19]: ./00_out/conv1_1.png "Grayscaling"
[image20]: ./00_out/conv3_1.png "Grayscaling"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?  --> 34799 sample
* The size of the validation set is ? --> 4410 sample
* The size of test set is ? --> 12630 sample
* The shape of a traffic sign image is ? --> (32, 32, 3)
* The number of unique classes/labels in the data set is ? --> 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it showed better results as the RGB and that could be as a reason of adding more wights to the first convolutional layer --> slower learning rates.

As a last step, I normalized the image data to help the optimizer finding the solution faster.

Here is an example of a traffic sign image before and after grayscaling and normalization.

![alt text][image2]

I decided to display/monitor the accuracy on both sets training and validation and that is to see if we suffer any kind of over-fitting, see the image below.
it shows a max validation accuracy at Epoch 43 which exceeds the 97% given that the training set accuracy was about 98%
![alt text][image9]

Additionaly I monitord the loss function with every Epoch.
![alt text][image10]


therefore I decided to generate additional data to come-over the under-fitting we suffering.

To add more data to the the data set, I tried the following techniques. 
- Adding guissen noise.
- Adding pepper noise.
- Bluring

I could not get the rotation results as it takes too much time.

Here is an example of an original image and an augmented image:

![alt text][image11]
![alt text][image12]
![alt text][image13]

The above additional samples could not add that much to the accuracy level so they were removed back.

To come-over the lighting problem within some of the samples the histogram equalizer technique is used.
it was clear that the adaptive equalizer is dealing better with images with high lighting nevertheless the overall performance from the normal equalizer (in our case) was better.

![alt text][image14]
![alt text][image15]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, Valid padding, outputs 28x28x6 	|
| RELU					|												|
| Dropout			    | 75%										    |
| Max pooling	      	| 2x2 stride,  outputs 14x14x6				    |
| Convolution 5x5     	| 1x1 stride, Valid padding, outputs 10x10x16 	|
| RELU					|												|
| Dropout			    | 75%										    |
| Max pooling	      	| 2x2 stride,  outputs 5x5x16				    |
| Fully connected		| 400*200       								|
| RELU					|												|
| Dropout			    | 50%										    |
| Fully connected		| 200*100       								|
| RELU					|												|
| Dropout			    | 50%										    |
| Fully connected		| 100*43       								|
| Softmax				|              									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following parameters
- Optimizer = AdamOptimizer
- EPOCHS = 50
- BATCH_SIZE = 32
- learning_rate = 0.001
- fc_keep_prob_value = .5
- conv_keep_prob_value = .75
- conv1_depth_val = 6
- conv2_depth_val = 16
- conv_filter_size = 5


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.
I started with the LeNet Architecture and I was targeting 98% for the validation set, firstly I started to tun the parameters batch size, epoch number to find-out the most convenient combinations, then I started to change the architecture by changing the width of different layers at first and then by adding additional layers (convolutional or fully connected). Adding the dropout out layers had a good impact also shuffling the data could improve the results.
I tried different optimizers including the momentum one, but i could not notice a remarkable change, then I started to have a look at the data, how to increase it and how to improve its quality.

My final model results were: -> may vary from run to another but in average like below 
* training set accuracy of ? 99,6%
* validation set accuracy of ? 97,5%
* test set accuracy of ? 94,3%

![alt text][image16]


![alt text][image1616]
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (120km/h)	| Speed limit (120km/h)							| 
| No vehicles  			| No vehicles									|
| Priority road			| Priority road									|
| Road work	      		| Road work					 				    |
| Vehicles over 3.5 metric tons prohibited			| Vehicles over 3.5 metric tons prohibited      							|


The model was able to correctly the five pictures correctly, see below the result from the testing script, it shows the 
classification and the label for each image and additionally the cross entropy for each.

![alt text][image17]

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

from the cost results above, it is clear that images with circular shape have higher cost and on the other side images with sharp edges "unique shapes" have lower cost functions. 


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Below is a sample from our first convolutional layer feature map.
![alt text][image19]
![alt text][image20]

