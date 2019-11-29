#!/usr/bin/env python

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
  
  # Returns an array of the x-coordinates for all blobs.
  def detect_blob_centre_xs(self, image):
    yX = self.detect_yellow_x(image)
    bX   = self.detect_blue_x(image)
    gX  = self.detect_green_x(image)
    rX    = self.detect_red_x(image)

    return np.array([yX, bX, gX, rX])

  # Detects the x component of a yellow blob's centre
  def detect_yellow_x(self, image):
    # thresholding the raw image on the colour yellow
    yellow_blob = cv2.inRange(image, (0, 100, 100), (0, 255, 255))
    # the kernel used for dilation
    kernel = np.ones((5, 5), np.uint8)
    # dilating the blob to better find the centre
    # 8 iterations are done to minimise the impact of occlusion by parts of
    # the robot arm
    yellow_blob = cv2.dilate(yellow_blob, kernel, iterations=8)
    # obtain the image moments of the blob
    M = cv2.moments(yellow_blob)
    # calculate the x-coordinate (measured in pixels) of the blob's centre
    cX = int(M['m10'] / M['m00'])
    return cX

  def detect_blue_x(self, image):
    blue_blob = cv2.inRange(image, (100, 0, 0), (255, 0, 0))
    kernel = np.ones((5, 5), np.uint8)
    blue_blob = cv2.dilate(blue_blob, kernel, iterations=8)
    M = cv2.moments(blue_blob)
    cX = int(M['m10'] / M['m00'])
    return cX

  def detect_green_x(self, image):
    green_blob = cv2.inRange(image, (0, 100, 0), (0, 255, 0))
    kernel = np.ones((5, 5), np.uint8)
    green_blob = cv2.dilate(green_blob, kernel, iterations=8)
    M = cv2.moments(green_blob)
    cX = int(M['m10'] / M['m00'])
    return cX

  def detect_red_x(self, image):
    red_blob = cv2.inRange(image, (0, 0, 100), (0, 0, 255))
    kernel = np.ones((5, 5), np.uint8)
    red_blob = cv2.dilate(red_blob, kernel, iterations=8)
    M = cv2.moments(red_blob)
    cX = int(M['m10'] / M['m00'])
    return cX

  # Defines publisher and subscriber
  def __init__(self):
    # initialize the node named image_processing
    rospy.init_node('image_processing', anonymous=True)
    # initialize a publisher to send images from camera2 to a topic named image_topic2
    self.image_pub2 = rospy.Publisher("image_topic2",Image, queue_size = 1)
    # initialize a publisher to publish x-coordinates for blob centres
    self.x_centres_pub = rospy.Publisher("x_centres_topic", Float64MultiArray, queue_size = 1)
    # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
    self.image_sub2 = rospy.Subscriber("/camera2/robot/image_raw",Image,self.callback2)
    # initialize the bridge between openCV and ROS
    self.bridge = CvBridge()


  # Recieve data, process it, and publish
  def callback2(self,data):
    # Recieve the image
    try:
      self.cv_image2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
    # Uncomment if you want to save the image
    #cv2.imwrite('image_copy.png', cv_image)
    #im2=cv2.imshow('window2', self.cv_image2)
    cv2.waitKey(1)

    # Publish the results
    try:
      self.image_pub2.publish(self.bridge.cv2_to_imgmsg(self.cv_image2, "bgr8"))
    except CvBridgeError as e:
      print(e)


    self.joints = Float64MultiArray()
    self.joints.data = self.detect_blob_centre_xs(self.cv_image2)
    self.x_centres_pub.publish(self.joints)

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
