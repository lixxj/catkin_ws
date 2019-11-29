#!/usr/bin/env python

import roslib, sys, rospy, cv2, math
import numpy as np
from std_msgs.msg import String, Float64MultiArray, Float64
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

############### COLOUR DECTECTION FUNCTIONS ###############  
### THE CODE ΙΝ ΤΗΕ FOLLOWING PART IS RETRIEVED FROM IVR_LABS
### AND IS FURTHER MODIFIED TO USE IN THIS ASSIGNMENT
def detect_blob(image, low, high):
    # Isolate the colour region
    mask = cv2.inRange(image, low, high)
    # Dilate the region 
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    # Obtain the moments of the region
    M = cv2.moments(mask)
    if M['m00'] == 0: # colour not found
	    return np.array([np.nan,np.nan])
    # Calculate pixel coordinates of the blob
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return np.array([cx, cy])

def detect_red(image):
    return detect_blob(image, (0, 0, 100), (15, 15, 255))

def detect_green(image):
    return detect_blob(image, (0, 100, 0), (15, 255, 15))

def detect_blue(image):
    return detect_blob(image, (100, 0, 0), (255, 15, 15))

def detect_yellow(image):
    return detect_blob(image, (0, 100, 100), (0, 255, 255))
############### END OF COLOUR DECTECTION FUNCTIONS ###############

def detect_joints(image):
    # Undistort the image
    center = (image.shape[1] / 2, image.shape[0] / 2)
    camera_matrix = np.array([[image.shape[1], 0, center[0]], [0, image.shape[1], center[1]], [0, 0, 1]], dtype="double")
    image = cv2.undistort(image, camera_matrix, (2, 2, 0, 0, 2))
    # Detect blobs (joints) by colour
    yellow = detect_yellow(image)
    blue = detect_blue(image)
    green = detect_green(image)
    red = detect_red(image)
    return yellow, blue, green, red

# Pixel to Meter conversion rate 
def pixel2meter(yellow, blue, green, red):
    # find the distances between two joints
    a1 = 2. / np.sqrt(np.sum((blue - yellow) ** 2))
    a2 = 3. / np.sqrt(np.sum((green - blue) ** 2))
    a3 = 2. / np.sqrt(np.sum((red - green) ** 2))
    
    return (a1 + a2 + a3) / 3 

def in_world_frame(im_f, image):
    base = detect_yellow(image)
    base_f = np.zeros(2)
    base_f[0] = im_f[0] - base[0]
    base_f[1] = -(im_f[1] - base[1])
    world_f = pixel2meter(*detect_joints(image)) * base_f
    return world_f

class image_converter:

    # Defines publisher and subscriber
    def __init__(self, sub, axis):
        # initialize a publisher to send joints' angular position to a topic called joints_pos
        self.joints_angle_pub = rospy.Publisher("image_processing/joints_angles_" + axis, Float64MultiArray, queue_size=10)
        self.pos_pub = rospy.Publisher("image_processing/joints_positions_" + axis, Float64MultiArray, queue_size=10)
        # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
        self.image_sub = rospy.Subscriber(sub, Image, self.callback)
        self.axis = axis
        # initialize the bridge between openCV and ROS
        self.bridge = CvBridge()

    # Calculate the relevant joint angles from the image
    def detect_joint_angles(self, image):
        yellow, blue, green, red = detect_joints(image)
        p2m = pixel2meter(yellow, blue, green, red)
        # Obtain the centre of each coloured blob
        center = p2m * yellow
        circle1Pos = p2m * blue
        circle2Pos = p2m * green
        circle3Pos = p2m * red
        # Solve using trigonometry
        ja1 = np.arctan2(center[0] - circle1Pos[0], center[1] - circle1Pos[1])
        ja2 = np.arctan2(circle1Pos[0] - circle2Pos[0], circle1Pos[1] - circle2Pos[1]) - ja1
        ja3 = np.arctan2(circle2Pos[0] - circle3Pos[0], circle2Pos[1] - circle3Pos[1]) - ja2 - ja1

        return np.array([ja1, ja2, ja3])

    # Recieve data, process it, and publish
    def callback(self, data):
        # Recieve the image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
            return
        if cv_image is None:
            rospy.log_warning("No image in image_processing_{}".format(self.axis))
            return

        # Calculate joint angles
        joint_angles = self.detect_joint_angles(cv_image)
        joints_msg = Float64MultiArray()
        joints_msg.data = joint_angles

        # Publish joint angles
        self.joints_angle_pub.publish(joints_msg)

        # End effector positions
        red_im = detect_red(cv_image)
        red_world = in_world_frame(red_im, cv_image)
        green_im = detect_green(cv_image)
        green_world = in_world_frame(green_im, cv_image)
        msg = Float64MultiArray()

        # publishes [rx, ry, rz, gx, gy, gz] where r is red, g is green joint
        if self.axis == "x":
            msg.data = [0, red_world[0], red_world[1], 0, green_world[0], green_world[1]]
        elif self.axis == "y":
            msg.data = [red_world[0], 0, red_world[1], green_world[0], 0, green_world[1]]

        self.pos_pub.publish(msg)

# call the classes
def main(args):
    rospy.init_node('image_processing')
    ic1 = image_converter("/camera1/robot/image_raw", "x")
    ic2 = image_converter("/camera2/robot/image_raw", "y")

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)
