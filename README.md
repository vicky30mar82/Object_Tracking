Link:
https://livecodestream.dev/post/object-tracking-with-opencv/


Object Tracking with OpenCV (Python)

Engineers and computer scientists have long been trying to allow computers to see and interpret visual data and perform certain 
functions based on the data collected. That is where the idea of computer vision arose. Computer vision aims to automate the 
processes of machines, which can be performed by human vision. You can say as artificial intelligence has given computers the 
ability to think, computer vision gives them the ability to see and interpret sight. Computer vision allows the machines to 
perform the functions of the human eye and mind, but instead of nerves and retinas, this has to be done with the help of cameras 
and algorithms.



For the past two decades, AI-driven computer vision has provided multiple methods to perform even a single function that a human 
brain can accomplish related to vision. The implementation of modern computer vision techniques has exponentially revolutionized 
technology. Now, it is being adapted in almost every other tech domain, be it medical diagnosis, autonomous cars, or background 
removal from images and videos. Moving on to the problem of object tracking, the following article probs the concept in a much deeper way. 
We start from the basics and will move on to the full implementation of an object tracking algorithm using OpenCV. The article is divided into the following sections:

What is object tracking?
Object tracking is one such application of computer vision where an object is detected in a video, otherwise interpreted as a set 
of frames, and the object’s trajectory is estimated. For instance, you have a video of a baseball match, and you want to track the 
ball’s location constantly throughout the video. Object tracking is the method of tracking the ball’s location across the screen in 
real-time by estimating its trajectory.



Object tracking, on an abstract level, can be done with either of the two approaches existing in it. One is called Single Object 
Tracking (SOT), and the other one, Multiple Object Tracking (MOT). As understood by the name itself, single object tracking is when 
only a single specific object is being tracked in a video or a set of frames. Similarly, multiple object tracking is when various 
objects are being tracked simultaneously within the same video or set of frames. The latter one, for obvious reasons, is far more 
complicated than the former. MOT poses the main difficulty in the interaction of multiple objects, to be tracked, with each other. 
Hence, models for SOT cannot be directly applied to MOT and leads to poor accuracy.


Object tracking has lately been extensively used in surveillance, security, traffic monitoring, anomaly detection, robot vision, 
and visual tracking. Visual tracking is an exciting application where the future position of an object in a video is estimated 
without inputting the rest of the video into the algorithm. It can be thought of as looking into the future.

Difficulties in Object Tracking
Despite being a beneficial method, not every market and/or process can afford to perform object tracking due to the fact that 
one of the crucial hurdles in training an object tracking model is the training and tracking speed. Tracking algorithms are 
expected and needed to detect and localize the object in a video in a fraction of a second and with high accuracy. This detection 
speed can be significantly tampered with involuntarily due to the variety of background distractions in any scenario. Another 
significant difficulty in object tracking is the variation in the spatial scales. An object can be present in an image (or a video) 

and in various sizes and orientations.

Another issue with object tracking, which is also a significant issue in object detection and recognition, is occlusion. 
Occlusion is when multiple objects come so close together that they appear to be merged. This can confuse the computer 
into thinking the merged object is a single object or simply wrongly identifying the object.


Aside from these, several issues pose difficulties in object tracking, such as switching of identity after crossing, 
motion blur, variation in the viewpoint, cluttering of similar objects in the background, low resolution, and variation in the illumination.

Object Tracking Approaches
Since now, many object tracking techniques have been developed, some for SOT, some for MOT, and some for both. 
These techniques include both classical computer vision-based architectures and also deep learning-based architectures. 
The most well-known methods and architectures for object tracking are as follows.

OpenCV-based object tracking
Object tracking using OpenCV is a popular method that is extensively used in the domain. OpenCV has a number of built-in 
functions specifically designed for the purpose of object tracking. Some object trackers in OpenCV include MIL, CSRT, GOTURN, and MediandFlow. 
Selecting a specific tracker depends on the application you are trying to design. Each tracker has its advantages and disadvantages, 
and a single type of tracker is not desired in all the applications.

MDNet
MDNet is short for Multi-Domain Convolutional Neural Network Tracker. It is a state-of-the-art visual tracker based on a convolutional 
neural network. It is also the winner of the VOT2015 challenge. It is composed of multiple shared layers and branches of domain-specific layers. 
The convolutional layers at the bottom of the layer stack learn the domain-independent features, and this feature extraction is shared across 
the whole video sequence. As for the top fully connected layer, it is unique for every frame, and it learns the features specific to the domain, 
i-e the high-level abstract features inherent to the particular frame of the video sequence it is being applied on. To learn more about MODNet, 
refer to Learning Multi-Domain Convolutional Neural Networks for Visual Tracking.

DeepSort
DeepSort is one of the most widely used object tracking architectures. It uses YOLO v3 for computing the bounding boxes around the 
objects in the videos. It is the extension of the (Simple Online and Realtime Tracking) SORT algorithm. It uses the kalman filter 
from the SORT algorithm and uses an identification model called ReID to link the bounding boxes with the estimated tracks of the objects. 
In case no ID matches the track, the object and the track are assigned a new ID. DeepSort allows tracking objects through more prolonged 
periods of occlusion. To further learn about DeepSort, visit Simple Online and Realtime Tracking with a Deep Association Metric . 
For the implementation of the algorithm, see its GitHub repository .

ROLO
Usage of the Long Short-Term Memory (LSTM) networks with the convolutional neural networks for object tracking. 
A famous example of the method is ROLO, which stands for Recurrent YOLO. You Only Look Once (YOLO) is very well-known 
object detection and recognition algorithm. ROLO uses YOLO for object detection and an LSTM for estimating 
the trajectory of the object. With the regression capability of LSTMs both spatially and temporally, ROLO can interpret 
a series of high-level visual features directly into the coordinates of tracked objects.


Many other approaches, apart from the ones above, have been developed for object tracking. In this article, 
we are going to dive into the process of using OpenCV for object tracking.

What is OpenCV?
OpenCV is a well-known open-source library that is primarily used for a variety of computer vision applications. 
It has also been widely used in machine learning, deep learning, and image processing. 
It helps in processing data containing images and videos. Since today, OpenCV has been used in several mainstream applications, 
including object detection and recognition, autonomous cars and robots, automated surveillance, anomaly detection, 
video and image search retrieval, medical image analysis, and object tracking. It can also be integrated with other 
libraries and can process array structures of libraries such as NumPy. It is an extensive library in both the sense of 
functionality and extensions; besides having an enormous toolbox of functions and algorithms; it supports not only Python but also 
C, C++, and Java. Moreover, it further supports Windows, Linux, ac OS, iOS, and Android.

Nowadays, OpenCV, being a library for computer vision, is the majority of the time used in artificial intelligence and 
its modern applications involving visual data such as images and videos. Various convolutional neural network-based 
architectures demand the support of OpenCV for both preprocessing and postprocessing. To learn more about OpenCV, 
refer to our study on Essential OpenCV Functions to Get You Started into Computer Vision.

How can OpenCV be used in object tracking?
OpenCV offers a number of pre-built algorithms developed explicitly for the purpose of object tracking. 
The following trackers are the available trackers in OpenCV:

BOOSTING Tracker:
BOOSTING tracker is based on the AdaBoost algorithm of machine learning. 
The classifier is to be trained at runtime learning on the positive and 
negative examples of the object to be tracked. It is over a decade old. 
It is slow and doesn’t work very well, even towards some relatively more superficial data.

MIL Tracker:
It is similar in concept to the BOOSTING tracker, with the only difference that instead 
of only using the current location of the object as a positive example for the classifier, 
it also looks into a small portion of the neighborhood of the thing. MIL tracker has better 
accuracy than BOOSTING, but it does a poor job of reporting failure.

KCF Tracker:
It stands for Kernelized Correlation Filters. KCF builds on the concept that multiple positive 
examples in a single bag of MIL tracker have large overlapping regions. The overlap gives rise 
to some intuitive mathematical approaches for the KCF tracker.


CSRT Tracker:
CSRT, otherwise known as Discriminative Correlation Filter with Channel and Spatial Reliability (DCF-CSR), 
used a spatial reliability map to adjust the filter to the part of the selected frame for tracking. 
This helps in the localization of the object of interest. It also gives high accuracy for comparatively lower fps (25 fps).

MedianFlow Tracker:
This tracker tracks both forward and backward displacements of an object in real-time and measures the error and 
difference between the two trajectories. Minimizing this error allows it to detect tracking failures and select the 
most reliable trajectories.

TLD Tracker:
It stands for Tracking, Learning, and Detection. This tracker follows the object frame by frame and localizes 
its position learned from the previous tracking, simultaneously correcting the tracker if necessary.

MOSSE Tracker:
It stands for Minimum Output Sum of Squared Error. It used an adaptive correlation for tracking purposes which outputs 
stable correlation filters. It is robust to scale, pose, non-rigid deformations, and lighting changes. 
It can also handle occlusion and can instantly resume the tracking when the object reappears. 
But on a performance scale, it lags deep earning based GOTURN.


GOTURN Tracker:
This is the only tracker based on a deep learning approach. It is developed using convolutional neural networks. 
It is accurate in that it is robust to deformations, lighting changes, and viewpoint changes; at the same time, 
the downside is that it cannot handle occlusion well.
