# adapted from https://github.com/nileshchopda/Traffic-Light-Detection-And-Color-Recognition

import numpy as np
import cv2
from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Int32, Int32MultiArray
import time
from time import strftime, localtime

path = "/home/thisiswooyeol/PycharmProjects/FMTC-SNU/'2023 S1'/traffic_light_detection"

class TrafficLightDecoder:
    def __init__(self):
        rospy.init_node("traffic_node", anonymous=True)
        self.bridge = CvBridge()
        self._img_sub = rospy.Subscriber("/vision/captured_image", CompressedImage,
                                         self._image_callback)
        self._bbox_sub = rospy.Subscriber("/vision/bbox/traffic_light", Int32MultiArray,
                                          self._bbox_callback)
        self._flag_pub = rospy.Publisher("/vision/stop_signal", Int32, queue_size=2)


    def _image_callback(self, msg):
        # Convert compressed image to cv2 image
        # ! self.rgb_img : shape=(480, 640, 3), dtype=uint8, ndarray, BGR encoding
        self.cap_img = self.bridge.compressed_imgmsg_to_cv2(msg)


    def _bbox_callback(self, bbox_coord_msg):
        self.bbox = bbox_coord_msg.data


    def crop_traffic_light_bbox(self, cap_img, bbox, plot_flag=False, save_flag=False):
        """
        From the YOLO bbox info, crop the traffic light image.

        Params
            cap_img : BGR encoded image input captured when YOLO starts inference
            bbox : xyxy format bbox coordinate from YOLO

        Return
            crop_img : BGR encoded cropped traffic light image
        """
        if len(bbox) == 0:  # if bbox is empty
            print("[Traffic Light Decoder] No traffic light detected")  # debug
            return None
        print("[Traffic Light Decoder] traffic light detected")  # debug

        # crop the traffic light image
        [xmin, ymin, xmax, ymax] = bbox
        crop_img = cap_img[ymin:ymax, xmin:xmax]

        if plot_flag is True:
            # debug whether the traffic light is cropped well
            cv2.imshow("[Traffic Light Decoder] Cropped Traffic Light", crop_img)
            cv2.waitKey(500)
        if save_flag is True:
            save_dir = strftime('%Y-%m-%d_%H:%M:%S',localtime())
            cv2.imwrite(f'{path}/output_images/yolo/YOLO_detected_{save_dir}.jpg', crop_img)

        return crop_img


    def decode_traffic_signal(self, crop_img, Threshold=0.05, plot_flag=False, save_flag=False):
        """
        Detect red and yellow color in the traffic light image as we need to stop the car when it detects these colors.
        HSV color space is used to detect the red and yellow color.
        If the percentage of red, yellow values in the image is greater than the threshold, then return `True`.
        
        Params
            crop_img : BGR encoded image input
            param Threshold : threshold for red and yellow detection

        Return
            bool(stop_flag): True if red or yellow is detected
        """        
        #### NEW IDEA ####
        # 1. Using Canny Edge Detection, detect the edge of all components in the image
        # 2. Using Hough Transform, detect the circle (traffic light) in the image
        # 3. Fill the circle with white color to make the circle as a mask
        # 4. Apply the mask to the original image
        # 5. Apply the red, yellow mask to the masked image
        # 6. Compare the percentage of red, yellow values in the masked image
        #    If the percentage of red, yellow values is greater than the threshold, then return True

        # START FROM HERE
        # 1. Canny Edge Detection & Morphological Closing
        # ! img_edge : single channel image
        img_edge = cv2.Canny(crop_img, 100, 200)
        img_edge = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, (3, 3))

        # 2. Hough Transform
        # [ERROR] : loop of ufunc does not support argument 0 of type NoneType which has no callable rint method
        # -> no circles detected
        # -> due to the HoughCircles function params: param1, param2 (there may be other multiple reasons...)
        # TODO : make a readable ERROR message description
        circles = cv2.HoughCircles(img_edge, cv2.HOUGH_GRADIENT,
                                   dp=1, minDist=20, param1=50, param2=20, minRadius=0, maxRadius=100)
        circles = np.uint16(np.around(circles))  # shape=(1, N, 3), dtype=uint16
        print(circles[0])  # debug

        # 3. Fill the circle with white
        roi_circle = np.zeros_like(crop_img[:, :, 0])  # black grayscale background
        for circle in circles[0]:  # circles[0].shape=(N, 3), N : number of circles
            center, radian = circle[:2], circle[2]
            cv2.circle(roi_circle, center, radian, 255, -1)

        # 4. Apply the mask to the original image
        # [ERROR] (-215:Assertion failed) (mtype == CV_8U || mtype == CV_8S) && _mask.sameSize(*psrc1)
        # in function 'binary_op' -> due to the mask needs to be single channel image
        circle_masked_img = cv2.bitwise_and(crop_img, crop_img, mask=roi_circle)

        if plot_flag is True:
            # debug whether the circle is detected well
            cv2.imshow("[Traffic Light Decoder] Circle Detected", circle_masked_img)
            cv2.waitKey(500)

        # 5. Apply the red, yellow mask to the masked image
        #  after applying mask to the image, use Morphological Opening to remove noise
        circle_masked_img_hsv = cv2.cvtColor(circle_masked_img, cv2.COLOR_BGR2HSV)  # ! convert to HSV color space
        red_yellow_roi = self.get_red_yellow_roi(circle_masked_img_hsv)
        red_yellow_img = cv2.bitwise_and(crop_img, crop_img, mask=red_yellow_roi)
        red_yellow_img = cv2.morphologyEx(red_yellow_img, cv2.MORPH_OPEN, (3, 3), iterations=1)

        # 6. Compare the percentage of red values
        rate = np.count_nonzero(red_yellow_img) / (crop_img.shape[0] * crop_img.shape[1])

        # plot the image for debugging
        if plot_flag is True:
            cv2.imshow("[Traffic Light Decoder] Red/Yellow Detected", red_yellow_img)
            cv2.waitKey(500)

        # save the image for debugging
        if save_flag is True: 
            save_dir = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime())

            if rate > Threshold:
                cv2.imwrite(f'{path}/output_images/stop/TL_STOP_{save_dir}_RAW.jpg', circle_masked_img)
                cv2.imwrite(f'{path}/output_images/stop/TL_STOP_{save_dir}.jpg', red_yellow_img)
            else:
                cv2.imwrite(f'{path}/output_images/go/TL_GO_{save_dir}.jpg', circle_masked_img)
                cv2.imwrite(f'{path}/output_images/go/TL_GO_{save_dir}_MASKED.jpg', red_yellow_img)

        if rate > Threshold:
            print("Red/Yellow Light Detected. Mask Rate :", rate)
            return True
        else:
            print("Red/Yellow Light Not Detected. Mask Rate :", rate)
            return False
    

    # apply red, yellow HSV mask to the image
    # ! Regarding the test environment, the HSV range should be adjusted.
    def get_red_yellow_roi(self, img):
        # lower red mask (0-10)
        lower_red = np.array([0, 128, 150])
        upper_red = np.array([10, 255, 255])
        mask0 = cv2.inRange(img, lower_red, upper_red)

        # upper red mask (170-180)
        lower_red1 = np.array([175, 128, 150])
        upper_red1 = np.array([179, 255, 255])
        mask1 = cv2.inRange(img, lower_red1, upper_red1)

        # defining the Range of yellow color
        lower_yellow = np.array([20, 128, 150])
        upper_yellow = np.array([35, 255, 255])
        mask2 = cv2.inRange(img, lower_yellow, upper_yellow)

        # red pixels' mask
        mask = mask0 | mask1 | mask2
        return mask

    
    # publish stop flag
    def traffic_flag_publish(self, stop_flag, log_flag=False):
        msg = Int32()
        msg.data = 1 if stop_flag else 0
        self._flag_pub.publish(msg)
        
        if log_flag:
            rospy.loginfo(msg)


if __name__ == "__main__":
    traffic_decoder = TrafficLightDecoder()
    rospy.sleep(1)  # wait for the node to be initialized

    rate = rospy.Rate(5)
    while not rospy.is_shutdown():
        print("\n==========================================================\n")
        inf_start = time.time()

        # get subscribed image and bbox info from YOLO node
        cap_img = traffic_decoder.cap_img
        bbox = traffic_decoder.bbox

        # detect traffic light and crop tl image
        crop_img = traffic_decoder(cap_img, bbox, plot_flag=False, save_flag=False)
        
        # decode traffic light signal
        try:
            stop_flag = traffic_decoder.decode_traffic_signal(crop_img, Threshold=0.05, plot_flag=False, save_flag=False)
        except Exception as e:
            print("[ERROR]", e)
            stop_flag = False
        
        # publish stop flag
        traffic_decoder.traffic_flag_publish(stop_flag, log_flag=True)

        print("TL judging time : ", (time.time() - inf_start) * 1000, "ms")
        print("\n==========================================================\n")

        rate.sleep()
