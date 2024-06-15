<!-- # Real-Time Object Detection and Avoidance in Robotics: Comparing Tiny-YOLOv3 and YOLOv3

Authors: Guillem Ribes Espurz (5229154), Ricardo Ramautar (6109217)

## Table of Contents
- [Abstract](#abstract)
- [Introduction](#introduction)
- [Motivation](#motivation)
  - [Training](#training)
  - [Robot Implementation](#robot-implementation)
- [Results](#results)
- [Discussion and Limitations](#discussion-and-limitations)
  - [Discussion](#discussion)
  - [Limitations](#limitations)
- [References](#references)
    
## Abstract

## Introduction
In numerous applications, robots must operate in unregulated and dynamic environments filled with moving obstacles. To navigate in these challenging environments, robots require obstacle avoidance systems to ensure they do not collide. For instance, self-driving vehicles need to account for other vehicles, pedestrians, and cyclists, whose actions can be unpredictable. Consequently, the obstacle avoidance systems in these self-driving vehicles must be able to account for sudden, sporatic behavior. 

Nowadays, most robots implement obstacle avoidance using LiDAR or ultrasound sensors instead of cameras. Although these sensors have the benefit of taking very fast measurements, the downside of using these sensors is that they abstract the world into simple distance measurements. This introduces an issue for some robotic applications for which the robot needs to react differently to different types of objects. For example, an autonomous vehicle must drive more dilligently when there is a person nearby, but can drive normally when it is simply driving past a trash can. Hence, for such applications object detection using stereo cameras can be implemented instead of using LiDAR or ultrasound to determine not only the position of obstacles, but also the class of obstacles.

## Motivation
Yet, object detection introduces its own set of problems. The main problem with object detection using neural networks is the speed-accuracy tradeoff. Very accurate object detection architectures tend to be very deep and therefore have slow inference, whereas more shallow architectures allow for faster inference, but are less accurate in detecting objects. This trade-off is exacerbated in robots, which often do not carry large GPUs, due to space, weight, or budget limits. Especially when using object detection for obstacle avoidance, inference needs to happen in real-time, which necesitates the need for an object detection architecture with fast inference, but still a sufficiently good accuracy for it to be useful in obstacle avoidance. 

There are however some object detection models that are specifically designed for use in end devices such as robots. Most notably, are the YOLO models which are designed to be fast during inference without the need for a powerful GPU. Although the newest version is YOLO v10, we decided to use the third version due to its compatibility with many different ROS packages.

To investigate the speed-accuracy tradeoff between shallow and deeper architectures, we decided to test both the standard YOLO v3 model and the YOLO v3-Tiny. YOLO v3 consists of 53 convolutional layers and is therefore quite deep. YOLO v3-Tiny is meant to be a much faster version of the standard YOLO v3 model by only having 13 convolutional layers and by implementing max pooling layers. [1] States a 4x speed improvement over YOLO v3, but also highlights a loss in accuracy. By comparing these two models, we hope to get a better understanding of what the actual speed-accuracy trade-off is between the two models on our robot to find out if the higher accuracy of a large model warrants the drop in inference speed.

Overall, we expect the standard YOLO v3 model to result in less false detections and provide more accurate bounding boxes compared to YOLO v3-Tiny. However, we expect YOLO v3-Tiny to result in a significantly higher detection frequency. 

Hence, the objective of this blog is to find out whether real-time object detection is feasible in small robots that do not carry large GPUs. This will be done by implementing object detection detection models into the Mirte Master robot. For the detection model, both standard YOLO v3 and YOLO v3-Tiny will be implemented to identify performance differences between a very small network (YOLO v3-Tiny) and a larger network (YOLO v3). The performance of the models will be expressed in the frame rate that can be achieved during inference and the accuracy of the detections in terms of F1 score. The reason for these metrics is that the frame rate and accuracy of the detections will play an important role in whether the object detection in the robot performs sufficiently well for application in for example obstacle avoidance.

## Implementation 

### Dataset
As part of another course, we got tasked by Lely with programming a barn cleaning robot. A new dataset containing images of manure and people was therefore created to train the models on. However, note that for praciticality, the manure was 3D printed. All the images in the dataset were made using the camera in the robot, such that the images are representative to what the robot can expect. Additionally, images were taken in various different environments with different surfaces and different lighting conditions to make the models robust to changes in environment. 

Approximately 600 images were taken. However, to increase the size of the dataset, these images were augmented by increasing and decreasing the brightness by 30%. In total, the complete dataset constituted 1783 images. 80% Of these images were randomly assigned to the training set, whereas the other 20% was assigned to the test set. 

### Training 
The training of the models was done remotely on a GPU. Both YOLO v3 and YOLO v3-Tiny were trained on the training set over 10.000 batches with a batch size of 64 images. The initial learning rate was set to 0.001 and was decreased by a factor of 10 every 1000 iterations. Additionally, the models were trained using Stochastic Gradient Descent with a momentum of 0.9 and a weight decay of a factor 0.0005. These parameters were taken from the standard .cfg file for training YOLO v3 and YOLO v3-Tiny.

### Method 
Once the model is trained the best weights are obtained in the format .weights and a .cfg file that denotes the architecture of the YOLO v3-tiny model. With those two files, a test set can be ran either locally or on the robot 

1. **Local Testing**: To test the model locally, the aforementioned weights and config files were used. The model was ran on a set of test images to evaluate its performance. This involved loading the model with the configuration and weights files, and then performing inference on the test images to obtain detection results (bounding boxes and class labels).

2. **Robot Deployment**: For real-time object detection on a robot, the model was ran on the robot itself. The process was similar to local testing but involved integrating the model into the robot’s software stack. This will be further explained in the next section. 

### Robot Implementation 
To run the model on the robot several things were done. 

Initially, the darknetROS repository was utilized, as it supports the integration of YOLO v3 models with ROS and requires only the implementation of the correct weights. By running YOLO v3-Tiny directly on the robot via darknetROS, we achieved an average frame rate of merely 0.2 FPS.

To improve performance, the darknet configuration was converted to ncnn, a high-performance neural network inference framework optimized for mobile platforms. By taking better advantage of the robot's hardware, a much greater frame rate could be achieved. 

#### Model Size Requirements and Limitations

One of the key factors to consider when implementing models on a robot is the size and complexity of the model. Larger models with more parameters generally require more computational power and memory, which can be a limitation for robots with limited hardware capabilities. For example, the robot used for this case is using an Orange Pi which has very limited computing power and thus can not run high end models in real time.

Implementing smaller, optimized models on robots has significant cost benefits. By avoiding the need for powerful and expensive GPUs, the overall cost of the robotic system can be reduced. This makes it feasible to deploy advanced object detection capabilities in a wider range of applications, from consumer robots to industrial automation systems.

<p>
    <img src="images/ObjectDetection.jpeg" alt>
    <em>Figure 1: Detections of YOLO v3 on some images of the test set.</em>
</p>

## Results
The accuracy of the trained models was tested on the test set, which was done on a laptop. Figure 1 shows some of the results of YOLO v3 on the test set. Consequently, Figure 2 shows the F1 score of both YOLO v3 (green) and YOLO v3-Tiny (blue) for a range of IoU thresholds. Additionally, the orange bars shows the difference between the two models. As can be seen from the graph, up until an IoU threshold of 50%, both models perform very well with an F1 score of approximately 0.99. This shows that both models are very effective in detecting the objects in question. However, from an IoU threshold of 50% onwards, the F1 scores of both models drop considerably. From this can be concluded that although effective in detecting objects, neither model is very accurate in setting the bounding boxes. Nonetheless, the graph also shows that YOLO v3 results in a noticeably higher F1 score for greater IoU scores compared to YOLO v3-Tiny, as also indicated by the orange bars. YOLO v3 is therefore more accurate in setting the bounding boxes than YOLO v3-Tiny.

| Model       | Frame Rate (FPS) | Standard Deviation (FPS) |
|-------------|------------------:|-------------------------:|
| YOLO v3      |              3.92 |                      0.47 |
| YOLO v3-Tiny |              6.53 |                      0.36 |

*Table 1: Average frame rate achieved by YOLO v3 and YOLO v3-Tiny*

However, YOLO v3-tiny achieves a considerably higher inference rate than YOLO v3. As shown in Table 1, YOLO v3 achieved an average inference rate of 3.92 with a standard deviation of $\pm$ 0.47, whereas YOLO v3-Tiny achieved an average inference rate of 6.53 with a standard deviation of $\pm$ 0.36.

When comparing the detections of YOLO v3 and YOLO v3-Tiny, we noticed that bounding boxes of YOLO v3 fit the object better, whereas the bounding boxes of YOLO v3-Tiny were slightly larger than the ground truth. However, for obstacle avoidance, this is not a problem. Hence, considering YOLO v3-Tiny offers comparable performance at lower IoU thresholds (around 5%-50%) while at the same time achieving faster inference, we argue that YOLO v3-Tiny is more ideal for real-time obstacle detection than YOLO v3.

When observing the detections at low IoU thresholds ($\leq 50$), we noticed that there were very few false positives and that the detections were overall very accurate. However, at lower IoU thresholds, the bounding boxes were often larger than the ground truths. Yet, the detected bounding boxes were more or less centered the same as the ground truths. Consequently, increasing the IoU threshold resulted in increasingly many false negatives. Thus, since false positives are not much of an issue an IoU threshold of 30% was selected when running on the robot with YOLO v3-Tiny. 

<p>
    <img src="images/f1_iou.png" alt>
    <em>Figure 2: Plot of F1 score as function of the IoU threshold achieved by the YOLO v3 (green) and YOLO v3-Tiny (blue) models on the test set.</em>
</p>

### Empiric observations
Video 1 shows the real-time detections of the trained YOLO v3-Tiny network on the robot with an IoU threshold of 30%. Note that the low frame rate is due to visualizing the bounding boxes, since the transmission of the frames from the robot to the laptop is quite slow. From the video can be observed that despite the low IoU threshold, there are no false positive detections. However, some false negative detections can be observed when the manure is far away. Additionally, the bounding boxes do not fit the detected objects very precisely. The bounding box for the person is for example slightly too big. This observation explains why the F1 score is quite small for large IoU thresholds and the F1 score is high for low thresholds. 

An important point to note is the fact that the model struggle to correctly detect manure when it is far away, resulting in a high number of false negatives. However, the detection accuracy improves significantly when the manure is closer, and the models detect it correctly all the way. This aspect is crucial to consider.

For the specific case of obstacle avoidance, this limitation is less problematic. Obstacle avoidance primarily requires accurate detection when the obstacle is near, rather than far away. Therefore, not being able to correctly classify objects at a distance is not a significant issue in this context.

However, for other applications that demand reliable object detection regardless of distance, this model may not be suitable. The tendency to miss distant objects could be a critical drawback in scenarios where detecting objects at all ranges is essential.

Another important observation is that sometimes an object (manure) is detected correctly at one frame and then not detected the next frame, but then detected again in consecutive frames. For application such as obstacle avoidance, this occurance can easily be solved using for example a kalman filter.


<p>
    <img src="images/yolo.gif" alt>
    <em>Video 1: Video of detections by the trained YOLO v3-Tiny on the robot with an IoU threshold of 30%. (The low frame rate is primarily due to the transmission of frames from the robot to the laptop)</em>
</p>


## Conclusion
The results have shown that due to the much smaller architecture of YOLO v3-Tiny, it is considerably faster than the standard YOLO v3. This speed advantage of YOLO v3-Tiny over YOLO v3 was found to be on average 2.61 FPS on our robot. 

However, as hypothesized, YOLO v3 performs detects the objects more precisely than YOLO v3-Tiny. What was surprising is that both models perform equally well in identifying objects. However, YOLO v3 is more precise in setting the bounding boxes compared to YOLO v3-Tiny.

## Discussion and Limitations

### Discussion
The goal of this blog was to find out whether real-time object detection is feasible in small robots that do not carry large GPUs. After implenting object detection models of varying sizes, we found that by implementing a small convolutional network such as YOLO v3-Tiny, an average frame rate of around 6.53 FPS can be achieved, while still having reasonably precise object detection. Implementing larger models like YOLO v3 results in a lower frame rate of 3.92, but generates more accurate bounding boxes. 

Although the object detection methods for detecting manure and people inside the robot were not applied for a practical application such as obstacle avoidance, we believe that the frame rate and accuracy achieved by both YOLO v3 and YOLO v3-Tiny are sufficient for obstacle avoidance in our robot, assuming not too high speeds. This is therefore something to be studied further. However, due to the only slight accuracy improvement that YOLO v3 offers over YOLO v3-Tiny and the much faster inference of YOLO v3-Tiny, we believe that YOLO v3-Tiny is more suitable for implementation in robots.

However, there are definitely some points of improvement to our research. Firstly, we made the mistake of labeling the images before splitting them into train and test set. Since RoboFlow was used for labeling, data augmentation was also immediately applied, which made it difficult to identify the unique image from their respective augmented images. Therefore, by randomly splitting the resulting dataset, the test set will contain images with identical counterparts with a change in brightness inside the training set. Hence, the F1-scores that were found may be exagerated and partly due to overfitting, since the test images are not totally obfuscated during training. However, our empiric findings of the performance of the object detection in the robots remain valid and support our claim that the models perform quite well.

Additionally, potentially a better detection performance could have been achieved for both models if the models were pre-trained on large object detection datasets such as COCO and subsequently fine-tuned on our custom dataset.

## References 
[1] https://ieeexplore.ieee.org/document/9074315 -->

# Training and Testing
The `Train_and_Test.ipynb` notebook is what we ran to train and test YOLO v3 and YOLO v3-Tiny.

To recreate what we did, download [this](https://drive.google.com/file/d/1lNFxCJUVNRtA3umNjn6aJpXLx2sqS4JH/view?usp=sharing) zip file containing all the necessary weights, configs, and the dataset that we created. Next, simply run `Train_and_Test.ipynb`.