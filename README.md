# Understanding-Traffic-Light-Signal-for-Intersection-Task
Part of a project for autonomous vehicle competition for college students (2023 미래형자동차 자율주행 SW 경진대회)


## Introduction.
Imagine we are driving a car. When we face an intersection that we are about to pass through, we check where is the traffic light and judge if we can go or not. To automate this procedure as a self-driving car, (1)it should detect the traffic light(object detection task) and also (2)decode the traffic light image to make a vehicle control. I first referred to the repo: [Traffic-Light-Detection-And-Color-Recognition](https://github.com/nileshchopda/Traffic-Light-Detection-And-Color-Recognition). In this repository, he proposed a 2 step approach that (1)Tensorflow's fastRCNN performs an object detection task and (2)image processing using a HSV color mask.

The proposed method showed good performance, but I found two areas for improvement. First, fastRCNN is itself a two-step algorithm, so using a one-step algorithm like YOLO could be more advantageous for real-time traffic light detection. Second, the existing method checks if all pixels in the bbox of the traffic light are within the red and yellow mask, which can lead to misclassification of traffic light signal, as in the output_10 image, where a yellow traffic light cover is misclassified as a stop sign.

[output_10](/assets/readme_images/ouput_10.png)

Therefore, in my approach, I used YOLOv8 (the latest model of YOLO, which is most widely used in practical object detection) for object detection task, and I selected the circular panel part where the light comes out from the traffic light as the region of interest for red and yellow masking and performed color recognition.


## How to get started
### Requirements
- [ROS noetic](http://wiki.ros.org/noetic/Installation/Ubuntu)
- [ROS usb-cam package](https://github.com/ros-drivers/usb_cam)
- [PyTorch](https://pytorch.org/get-started/locally/)
- [YOLO](https://github.com/ultralytics/ultralytics)

### Installation

1. Clone this repository and move it to your own workspace.
```bash
git clone git@github.com:thisisWooyeol/Understanding-Traffic-Light-Signal-for-Intersection-Task.git
mv Understanding-Traffic-Light-Signal-for-Intersection-Task  YOUR-OWN-WS-PATH/src/
```

2. Move `usb_cam.yml` file to your `usb-cam/config/` folder and update your workspace.
```bash
# Note that cloned usb-cam package should be also placed in your own workspace folder
mv Understanding-Traffic-Light-Signal-for-Intersection-Task/usb_cam.yml usb-cam/config
source source YOUR-OWN-WS-PATH/devel/setup.sh
```

### If you want to finetune YOLO

1. Gather your custom images and its bbox label(.txt file) and split them into train and validation sets.

2. Edit the path for train and validation sets in `src/utils/data.yaml`.

3. run `tuning_yolo.py`.

### Run traffic light decoder

With your usb camera connected, run the `traffic_node` with launchfile
```bash
roslaunch usb_cam usb_cam.launch
roslaunch traffic-light-task TL_decoder-test.launch
```


## Key idea for color recognition task

### Summary
1. Using Canny Edge Detection, detect the edge of all components in the image.
2. Using Hough Transform, detect the circle (traffic light) in the image.
3. Fill the circle with white color to make the circle as a mask.
4. Apply the mask to the original image.
5. Apply the red, yellow mask to the masked image.
6. Compare the percentage of red, yellow values in the masked image. <br>
   If the percentage of red, yellow values is greater than the threshold, then return True

### With implemented codes
TBD
