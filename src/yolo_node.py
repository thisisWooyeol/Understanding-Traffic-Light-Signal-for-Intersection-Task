from ultralytics import YOLO
import torch
import cv2
from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Int32MultiArray
import copy
from time import strftime, localtime

# ! IMPORTANT : change the path to your own path
# If you want to save the detected image, then make the directory 'output_images/yolo' in the path
path = "."

class YoloDetection:
    def __init__(self, weights_path='../weights/best_0803.pt'):
        self.load_model(weights_path)

        rospy.init_node('yolo_detection', anonymous=True)
        self.bridge = CvBridge()
        self._bgr_sub = rospy.Subscriber('/usb_cam/image_raw/compressed', CompressedImage, 
                                         self._image_callback)
        self._img_pub = rospy.Publisher('/vision/captured_image', CompressedImage, queue_size=2)
        self._tl_bbox_pub = rospy.Publisher('/vision/bbox/traffic_light', Int32MultiArray, queue_size=2)
        self._cw_bbox_pub = rospy.Publisher('/vision/bbox/crosswalk', Int32MultiArray, queue_size=2)


    def load_model(self, weights_path):
        # fine-tuned with Traffic Light(class 9) and CrossWalk(class 0) dataset
        self.model = YOLO(weights_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)


    def _image_callback(self, msg):
        # Convert compressed image to cv2 image
        # ! self.bgr_img : shape=(480, 640, 3), dtype=uint8, ndarray, BGR encoding
        self.bgr_img = self.bridge.compressed_imgmsg_to_cv2(msg)


    def __call__(self,plot_flag=False, save_flag=False):
        """
        Detect COCO dataset objects from the image with fine-tuned YOLOv8 small model.
        Fine-tuning is done with Traffic Light(class 9) and CrossWalk(class 0) dataset.

        Params
            plot_flag : if True, then plot the detected image
            save_flag : if True, then save the detected image
        
        Return
            cap_img : BGR encoded image input
            bbox_info : class 'ultralytics.engine.results.Boxes'
        """
        with torch.no_grad():
            # Convert image to tensor
            cap_img = copy.deepcopy(self.bgr_img) # ! IMPORTANT : prevent the image from being updated

            # Run object detection
            # results : class 'ultralytics.engine.results.Results'
            print("[Yolo Node] YOLO detected") 
            results = self.model(cap_img)[0]
            bbox_info = results.boxes

            # Code for debugging
            plots = results.plot()
            if plot_flag is True:
                # debug whether yolo detects the traffic light well
                cv2.imshow("[Yolo Detection] YOLO detected", plots)
                cv2.waitKey(500)
            if save_flag is True:
                save_dir = strftime('%Y-%m-%d_%H:%M:%S',localtime())
                cv2.imwrite(f'{path}/output_images/yolo/YOLO_detected_{save_dir}.jpg', plots)

        return cap_img, bbox_info
    

    def filter_bbox(self, cap_img, bbox_info, min_score_thresh_tl=0.5, min_score_thresh_cw=0.5, 
                                                traffic_light_label=9, crosswalk_label=0):
        """
        Filter the bbox with score threshold and class label that we are interested in 
        (traffic light, crosswalk)
        
        Params
            cap_img : BGR encoded image input
            bbox_info : class 'ultralytics.engine.results.Boxes'
            min_score_thresh_tl : minimum IoU score threshold for traffic light
            min_score_thresh_cw : minimum IoU score threshold for crosswalk
            traffic_light_label : class label for traffic light
            crosswalk_label : class label for crosswalk

        Publish
            bbox_msg : bbox coordinates list for traffic light or crosswalk
            img_msg : compressed image for decoding the color of traffic light
        """
        boxes = bbox_info.xyxy.type(torch.int)
        scores = bbox_info.conf
        classes = bbox_info.cls

        # if no traffic light or crosswalk detected, publish empty array
        if boxes.shape[0] == 0:
            print("[YOLO node] no traffic light or crosswalk detected")
            topic_to_publish = None
            self.publish_bbox(topic_to_publish, [])
            self.publish_img(cap_img)

        # if traffic light or crosswalk detected, publish the bbox coordinates list
        for i in range(boxes.shape[0]):  # boxes.shape[0] : number of boxes
            if scores[i] > min_score_thresh_tl and classes[i] == traffic_light_label:
                print("[YOLO node] traffic light detected")
                topic_to_publish = 'TL'
            elif scores[i] > min_score_thresh_cw and classes[i] == crosswalk_label:
                print("[YOLO node] crosswalk detected")
                topic_to_publish = 'CW'

            # for bbox of traffic light and crosswalk, publish the xyxy bbox coordinates list
            bbox = [pixel_val.item() for pixel_val in boxes[i]]  # ! each item dtype is float32 not torch.float
            self.publish_bbox(topic_to_publish, bbox)
            # publish the captured image
            self.publish_img(cap_img)


    def publish_img(self, cap_img):
        # Convert cv2 image to compressed image
        msg = self.bridge.cv2_to_compressed_imgmsg(cap_img)
        self._img_pub.publish(msg)


    def publish_bbox(self, topic_to_publish, bbox_for_publish):
        # Convert bbox info to Int32MultiArray
        bbox_msg = Int32MultiArray()
        bbox_msg.data += bbox_for_publish

        # publish empty array to clear the previous bbox
        if topic_to_publish is None:
            self._tl_bbox_pub.publish(Int32MultiArray(data=[]))
            self._cw_bbox_pub.publish(Int32MultiArray(data=[]))
        elif topic_to_publish == 'TL':
            self._tl_bbox_pub.publish(bbox_msg)
            self._cw_bbox_pub.publish(Int32MultiArray(data=[]))
        elif topic_to_publish == 'CW':
            self._tl_bbox_pub.publish(Int32MultiArray(data=[]))
            self._cw_bbox_pub.publish(bbox_msg)


if __name__ == "__main__":   
    yolo_detection = YoloDetection()
    rospy.sleep(1)  # wait for the node to be initialized

    rate = rospy.Rate(5)    
    while not rospy.is_shutdown():
        cap_img, bbox_info = yolo_detection(plot_flag=True, save_flag=False)
        yolo_detection.filter_bbox(cap_img, bbox_info)
    
        rate.sleep()
