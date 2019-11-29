#!/usr/bin/env python2.7

import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError


class image_converter:
    # Defines publisher and subscriber
    def __init__(self):
        # initialize the node named image_processing
        rospy.init_node('image_processing', anonymous=True)
        # initialize a publisher to send images from camera2 to a topic named image_topic2
        self.image_pub2 = rospy.Publisher("image_topic2", Image, queue_size=10)
        # initialize a publisher to send joint position detected by camera 2
        self.pos_pub2 = rospy.Publisher('joint_pos2', Float64MultiArray, queue_size=5)
        # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
        self.image_sub2 = rospy.Subscriber("/camera2/robot/image_raw", Image, self.callback2)
        # initialize the bridge between openCV and ROS
        self.bridge = CvBridge()


    def detect_target(self,image):
        mask = cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), (11, 43, 46), (25, 255, 255))
        # This applies a dilate that makes the binary region larger (the more iterations the larger it becomes)
        kernel = np.ones((8, 8), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # if the target is hidden, then make it to the centre of red ball
        if np.where(mask != 0)[0].shape[0] == 0:
            return self.detect_red(image)
        # Obtain the moments of the binary image
        M = cv2.moments(mask)
        # Calculate pixel coordinates for the centre of the blob
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return [cx, cy]

    # Detecting the centre of the red circle
    def detect_red(self, image):
        # Isolate the red colour in the image as a binary image
        mask = cv2.inRange(image, (0, 0, 100), (0, 0, 255))
        # This applies a dilate that makes the binary region larger (the more iterations the larger it becomes)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        # Obtain the moments of the binary image
        M = cv2.moments(mask)
        # If the joint is shield by another one, return zero to indicate the overlapping
        if(M['m00'] == 0):
            return [0,0]
        # Calculate pixel coordinates for the centre of the blob
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return [cx, cy]

    # Detecting the centre of the green circle
    def detect_green(self, image):
        mask = cv2.inRange(image, (0, 100, 0), (0, 255, 0))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        M = cv2.moments(mask)
        if(M['m00'] == 0):
            return [0,0]
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return [cx, cy]

    # Detecting the centre of the blue circle
    def detect_blue(self, image):
        mask = cv2.inRange(image, (100, 0, 0), (255, 0, 0))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        M = cv2.moments(mask)
        if(M['m00'] == 0):
            return [0,0]
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return [cx, cy]

    # Detecting the centre of the yellow circle
    def detect_yellow(self, image):
        mask = cv2.inRange(image, (0, 100, 100), (0, 255, 139))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        M = cv2.moments(mask)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return [cx, cy]

    # Calculate the conversion from pixel to meter
    def pixel2meter(self, image):
        # Obtain the centre of each coloured blob
        circle1Pos = np.array(self.detect_blue(image))
        circle2Pos = np.array(self.detect_yellow(image))
        # find the distance between two circles
        dist = np.sum((circle1Pos - circle2Pos)**2)
        return 2 / np.sqrt(dist)

    # Calculate the relevant joint angles from the image
    def detect_joint_pos(self, image):
        a = self.pixel2meter(image)
        # Obtain the centre of each coloured blob
        center = self.detect_yellow(image)
        # Calculate the relative position of each joint with respect to the yellow joint
        circle1Pos = [j-i for i, j in zip(self.detect_blue(image), center)]
        circle2Pos = [j-i for i, j in zip(self.detect_green(image), center)]
        circle3Pos = [j-i for i, j in zip(self.detect_red(image), center)]
        target = [j-i for i, j in zip(self.detect_target(image), center)]

        return a * np.array(circle1Pos + circle2Pos + circle3Pos + target)

    # Recieve data, process it, and publish
    def callback2(self, data):
        # Recieve the image
        try:
            self.cv_image2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        # Uncomment if you want to save the image
        # cv2.imwrite('image2_copy.png', self.cv_image2)
        im2=cv2.imshow('window2', self.cv_image2)
        cv2.waitKey(1)

        # Detect the relative joint position in terms of the yellow joint
        joints_pos_data = self.detect_joint_pos(self.cv_image2)
        # Filter out extreme values which is likely to be wrong
        if(any([np.absolute(x) > 10 for x in joints_pos_data])):
            return

        self.joints_pos = Float64MultiArray()
        self.joints_pos.data = joints_pos_data

        # Publish the results
        try:
            self.image_pub2.publish(self.bridge.cv2_to_imgmsg(self.cv_image2, "bgr8"))
            self.pos_pub2.publish(self.joints_pos)
        except CvBridgeError as e:
            print(e)

# call the class
def main(args):
    ic = image_converter()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)
