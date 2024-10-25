# Gesture-Recognition
Deep Learning - Gesture Recognition
Deep Learning Course Project- Gesture Recognition
Problem Statement
As a data scientist at a home electronics company which manufactures state of the art smart televisions. We want to develop a cool feature in the smart-TV that can recognise five different gestures performed by the user which will help users control the TV without using a remote. 
•	Thumbs up		:  Increase the volume.
•	Thumbs down		: Decrease the volume.
•	Left swipe		: 'Jump' backwards 10 seconds.
•	Right swipe		: 'Jump' forward 10 seconds. 
•	Stop			: Pause the movie. 

Here’s the data: https://drive.google.com/uc?id=1ehyrYBQ5rbQQe6yL4XbLWe3FMvuVUGiL


Understanding the Dataset
The training data consists of a few hundred videos categorized into one of the five classes. Each video (typically 2-3 seconds long) is divided into a sequence of 30 frames (images). These videos have been recorded by various people performing one of the five gestures in front of a webcam - similar to what the smart TV will use. 
 

Objective
Our task is to train different models on the 'train' folder to predict the action performed in each sequence or video and which performs well on the 'val' folder as well. 


Two types of architectures suggested for analysing videos using deep learning:
1.	3D Convolutional Neural Networks (Conv3D)

3D convolutions are a natural extension to the 2D convolutions you are already familiar with. Just like in 2D conv, you move the filter in two directions (x and y), in 3D conv, you move the filter in three directions (x, y and z). In this case, the input to a 3D conv is a video (which is a sequence of 30 RGB images). If we assume that the shape of each image is 100 x 100 x 3, for example, the video becomes a 4D tensor of shape 100 x 100 x 3 x 30 which can be written as (100 x 100 x 30) x 3 where 3 is the number of channels. Hence, deriving the analogy from 2D convolutions where a 2D kernel/filter (a square filter) is represented as (f x f) x c where f is filter size and c is the number of channels, a 3D kernel/filter (a 'cubic' filter) is represented as (f x f x f) x c (here c = 3 since the input images have three channels). This cubic filter will now '3D-convolve' on each of the three channels of the (100 x 100 x 30) tensor
.
            

   	Figure 1: A simple representation of working of a 3D-CNN

2.	CNN + RNN architecture 

The conv2D network will extract a feature vector for each image, and a sequence of these feature vectors is then fed to an RNN-based network. The output of the RNN is a regular softmax (for a classification problem such as this one).

 
Figure 2: A simple representation of an ensembled CNN+LSTM Architecture

Data Generator
This is one of the most important part of the code. In the generator, we are going to pre-process the images as we have images of 2 different dimensions (360 x 360 and 120 x 160) as well as create a batch of video frames. The generator should be able to take a batch of videos as input without any error. Steps like cropping, resizing and normalization should be performed successfully.

Data Pre-processing

•	Resizing and cropping of the images. This was mainly done to ensure that the NN only recognizes the gestures effectively rather than focusing on the other background noise present in the image.
•	Normalization of the images. Normalizing the RGB values of an image can at times be a simple and effective way to get rid of distortions caused by lights and shadows in an image.
•	At the later stages for improving the model’s accuracy, we have also made use of data augmentation, where we have slightly rotated the pre-processed images of the gestures in order to bring in more data for the model to train on and to make it more generalizable in nature as sometimes the positioning of the hand won’t necessarily be within the camera frame always.
 
NN Architecture development and training
•	Experimented with different model configurations and hyper-parameters and various iterations and combinations of batch sizes, image dimensions, filter sizes, padding and stride length were experimented with. We also played around with different learning rates and ReduceLROnPlateau was used to decrease the learning rate if the monitored metrics (val_loss) remains unchanged in between epochs.
•	Experimented with SGD() and Adam() optimizers but went forward with Adam() as it lead to improvement in model’s accuracy by rectifying high variance in the model’s parameters.  
•	Made use of Batch Normalization, pooling and dropout layers when our model started to overfit, this could be easily witnessed when our model started giving poor validation accuracy inspite of having good training accuracy. 
•	Early stopping was used to put a halt at the training process when the val_loss would start to saturate / model’s performance would stop improving.

Observations
•	It was observed that as the Number of trainable parameters increase, the model takes much more time for training.
•	Increasing the batch size greatly reduces the training time but this also has a negative impact on the model accuracy. So, there is always a trade-off here on basis of priority -> If we want our model to be ready in a shorter time span, choose larger batch size else one should choose lower batch size if you want your model to be more accurate.
•	Data Augmentation and Early stopping greatly helped in overcoming the problem of overfitting which our initial version of model was facing. 
•	Conv3D based model had better performance than CNN+LSTM based model. This is something which depends on the kind of data we used, the architecture we developed and the hyper-parameters we chose.
•	Transfer learning boosted the overall accuracy of the model. MobileNet Architecture was used due to it’s light weight design and high speed performance coupled with low maintenance as compared to other well-known architectures like VGG16, AlexNet, GoogleNet etc. 
•	For detailed information on the Observations and Inference, please refer below table.

 
 
 
Table 1: Observations and Results for all models

Finalizing Model
Considering the accuracy, memory footprint and number of parameters of all the models experimented, Model 8 – Conv3D Model is finalized due to the below reasons.
Reasons:
	Training Accuracy: 89%, Validation Accuracy: 89%)

	Number of Parameters = 230,949, significantly less as compared to other model’s parameters vs performance
	Learning rate gradually decreasing after some Epochs
	Model size = 2.75 MB (h5 file “model-00028-0.31119-0.89517-0.38711-0.89000.h5” attached below)

 

Model 19 – Transfer Learning with GRU and training all weights also performed extremely well (Training accuracy: 99% and Validation accuracy 100%). However, the model has 3,693,253 parameters and h5 file size is 42.4 MB. Model 8 is preferred over model 19 as model 8 is able to give good accuracy on the least number of parameters so that it can fit in the memory of the webcam.


