import rospy
from std_msgs.msg import Int32, Int32MultiArray
import time

class CrosswalkDetection:
    def __init__(self):
        self.crosswalk_distance = None

        rospy.init_node('crosswalk_detection', anonymous=True)
        self._bbox_sub = rospy.Subscriber('/vision/bbox/crosswalk', Int32MultiArray,
                                          self._bbox_callback)
        self._cw_distance_pub = rospy.Publisher('/vision/crosswalk_distance', Int32, queue_size=2)


    def _bbox_callback(self, bbox_coord_msg):
        self.bbox = bbox_coord_msg.data
    

    def is_crosswalk_in_xbound(self, box_xyxy, bbox_center_bound=[0.1, 0.9], img_width=640):
        # check whether the crosswalk is close to the car
        # by if the center of crosswalk bbox is in the box_position_threshold
        # box_position_threshold : [xmin, ymin, xmax, ymax]
        # TODO: adjust the center bound considering your environment
        [xmin, _, xmax, _] = box_xyxy
        box_center_x = (xmin + xmax) / 2
        
        if bbox_center_bound[0]*img_width < box_center_x < bbox_center_bound[1]*img_width:
            return True
        else:
            return False


    def crosswalk_y_distance(self, box_xyxy, img_height=480):
        # calculate the distance between the crosswalk and the car
        # by the center of crosswalk bbox
        [_, ymin, _, ymax] = box_xyxy
        box_distance_from_car = img_height - (ymin + ymax) / 2  # as y axis is flipped
        
        return round(box_distance_from_car)
    

    def publish_crosswalk_distance(self, distance):
        msg = Int32()
        msg.data = distance
        self._cw_distance_pub.publish(msg)

        print("[Crosswalk Detection] Publish crosswalk distance : ", distance)
        rospy.loginfo(msg)
    

if __name__ == '__main__':
    cw_detection = CrosswalkDetection()
    rospy.sleep(1) # wait for the node to be initialized

    rate = rospy.Rate(5)
    while not rospy.is_shutdown():
        # by using bbox info from YOLO, detect crosswalk
        print("==========================================================")
        inf_start = time.time()
        cw_detection.crosswalk_distance = None  # reset distance of crosswalk

        bbox_xyxy = cw_detection.bbox
        # if there is no crosswalk bbox, skip
        if len(bbox_xyxy) != 0 and cw_detection.is_crosswalk_in_xbound(bbox_xyxy):
            cw_detection.crosswalk_distance = cw_detection.crosswalk_y_distance(bbox_xyxy)

        if cw_detection.crosswalk_distance is not None:
            cw_detection.publish_crosswalk_distance(cw_detection.crosswalk_distance)
        else:
            print("[Crosswalk Detection] No crosswalk detected")
            cw_detection.publish_crosswalk_distance(-1)

        print("CW judging time : ", (time.time() - inf_start) * 1000, "ms")
        print("\n==========================================================\n")

        rate.sleep()