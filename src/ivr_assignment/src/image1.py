#!/usr/bin/env python

# rostopic pub -1 /robot/joint2_position_controller/command std_msgs/Float64 "data: 1.0"

import math
import time
import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError
import pyximport; pyximport.install()
import cython_functions

class image_converter:

    # Defines publisher and subscriber
    def __init__(self):
        self.last_known_locations = np.zeros(shape=(5,3))
        self.my_estimation = (-1, -1, -1, -1)

        # initialize the node named image_processing
        rospy.init_node('image_processing', anonymous=True)
        # initialize a publisher to send images from camera1 to a topic named image_topic1
        self.image_pub1 = rospy.Publisher("image_topic1", Image, queue_size=1)
        # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
        #self.image_sub1 = rospy.Subscriber("/camera1/robot/image_raw", Image, self.callback1, queue_size=1, buff_size=2**24)
        self.image_sub1 = rospy.Subscriber("/camera1/robot/image_raw", Image, self.callback1)

        # initialize a publisher to send images from camera2 to a topic named image_topic2
        self.image_pub2 = rospy.Publisher("image_topic2", Image, queue_size=1)
        # initialize a subscriber to recieve messages rom a topic named /robot/camera2/image_raw and use callback function to recieve data
        #self.image_sub2 = rospy.Subscriber("/camera2/robot/image_raw", Image, self.callback2, queue_size=1, buff_size=2**24)
        self.image_sub2 = rospy.Subscriber("/camera2/robot/image_raw", Image, self.callback2)

        # initialize the bridge between openCV and ROS
        self.bridge = CvBridge()

        # initialize time variables for the closed control part

        self.time_previous_step = np.array([rospy.get_time()], dtype='float64')
        self.time_previous_step2 = np.array([rospy.get_time()], dtype='float64')
        # initialize error and derivative of error for trajectory tracking
        self.error = np.array([[0.0,0.0,0.0]], dtype='float64')
        self.d_error = np.array([[0.0,0.0,0.0]], dtype='float64')
        self.d2_error = np.array([[0.0,0.0,0.0]], dtype='float64')

        #initialize publisher to send data about measured pos of the first target
        self.target_x_pub = rospy.Publisher("target_x", Float64, queue_size=1)
        self.target_y_pub = rospy.Publisher("target_y", Float64, queue_size=1)
        self.target_z_pub = rospy.Publisher("target_z", Float64, queue_size=1)

        #initialize publisher to send data about calculated pos of end effector by FK
        self.end_effector_x_FK = rospy.Publisher("FKend_effector_x", Float64, queue_size=1)
        self.end_effector_y_FK = rospy.Publisher("FKend_effector_y", Float64, queue_size=1)
        self.end_effector_z_FK = rospy.Publisher("FKend_effector_z", Float64, queue_size=1)

        #initialize publisher to send desired joint angles
        self.robot_joint1_pub = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size=1)
        self.robot_joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=1)
        self.robot_joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=1)
        self.robot_joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=1)

        #initialize the angles command
        self.q_d = np.array([0,1,0,0])

        self.DD_pub = rospy.Publisher("DD", Float64, queue_size=1)

    # Estimates the positions in 2D space of all the targets.
    def get_positions(self, img):
        img = img.copy()
        cython_functions.remove_greyscale(img)
        cython_functions.saturate(img)

        # The colour specified.
        yellow = np.array([0, 255, 255])
        blue = np.array([255, 0, 0])
        green = np.array([0, 255, 0])
        red = np.array([0, 0, 255])
        orange = np.array([108, 196, 253])

        # Process the images to extract only the desired colours.
        targets = [
            self.process_colour(img, yellow, 5),
            self.process_colour(img, blue, 5),
            self.process_colour(img, green, 5),
            self.process_colour(img, red, 5),
            self.process_orange(img, orange, 16)
        ]

        positions = [0, 0, 0, 0, 0]

        # Find the average coordinates of the images.
        for i in range(5):
            M = cv2.moments(targets[i])
            if M['m00'] == 0:
                positions[i] = (-1,-1)
                # print("Warning. Sphere " + str(i) + " has been lost in image " + str(img_num))
                continue

            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            positions[i] = (cx, cy)

        return positions

    # Processes a provided image to a grayscale image using the provided colours and colour thresholds.
    def process_colour(self, img, colour, colour_threshold):
        return self.greyscale(cython_functions.select_colour(img.copy(), colour, colour_threshold))

    # Makes an image greyscale.
    def greyscale(self, img):
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2GRAY)

    # Very similar to process_colour, but also removes thin parts of the image to remove the rectangle.
    def process_orange(self, img, colour, colour_threshold):
        img = cython_functions.select_colour(img.copy(), colour, colour_threshold)

        # Removing all thin (number of lit pixels in a row < 10) sections of the image
        # is a very effective way of removing all traces of the rectangle while still maintaining as much
        # of the circle as possible
        cython_functions.remove_thin_bits(img, 10, 2)

        return self.greyscale(img)

    # A debug function for visualising the calculated target positions
    def draw_spot_at(self, img, pos):
        try:
            img[pos[1], pos[0]] = [255, 255, 255]
            img[pos[1]+1, pos[0]] = [0, 0, 0]
            img[pos[1]-1, pos[0]] = [0, 0, 0]
            img[pos[1], pos[0]+1] = [0, 0, 0]
            img[pos[1], pos[0]-1] = [0, 0, 0]
        except:
            return

    #calculate the forward kinematics equations, q is an array of the input angles, return the coordinates of the end effector
    def ForwardK (self, q):
        end_effector = np.array([2*(np.sin(q[0])*np.sin(q[1])*np.cos(q[2]) + np.sin(q[2])*np.cos(q[0]))*np.cos(q[3]) + 2*np.sin(q[0])*np.cos(q[1])*np.sin(q[3]) + 3*(np.sin(q[0])*np.sin(q[1])*np.cos(q[2]) + np.sin(q[2])*np.cos(q[0])) ,-2*(np.cos(q[0])*np.sin(q[1])*np.cos(q[2]) - np.sin(q[0])*np.sin(q[2]))*np.cos(q[3]) - 2*np.cos(q[0])*np.cos(q[1])*np.sin(q[3]) + 3*(-np.cos(q[0])*np.sin(q[1])*np.cos(q[2]) + np.sin(q[2])*np.sin(q[0])) ,2*np.cos(q[1])*np.cos(q[2])*np.cos(q[3]) - 2*np.sin(q[1])*np.sin(q[3]) + 3*np.cos(q[2])*np.cos(q[1]) + 2])
        return end_effector


     # Calculate the robot Jacobian
    def calculate_jacobian(self,q):
        #because it is very big matrix, we chose to simplify by writing columns by columns : [J1, J2, J3, J4]
        J1 = np.array([2*np.cos(q[3])*(np.cos(q[0])*np.sin(q[1])*np.cos(q[2]) - np.sin(q[2])*np.sin(q[0])) + 2*np.cos(q[0])*np.cos(q[1])*np.sin(q[3]) + 3*(np.cos(q[0])*np.sin(q[1])*np.cos(q[2]) - np.sin(q[2])*np.sin(q[0]))
                    ,2*(np.cos(q[3])*np.sin(q[0])*np.sin(q[1])*np.cos(q[2]) + np.sin(q[0])*np.cos(q[1])*np.sin(q[3]) + np.cos(q[0])*np.sin(q[2])*np.cos(q[3])) + 3*(np.sin(q[0])*np.sin(q[1])*np.cos(q[2]) + np.cos(q[0])*np.sin(q[2]))
                    ,0])
        J2 = np.array([2*np.cos(q[3])*np.sin(q[0])*np.cos(q[1])*np.cos(q[2]) - 2*np.sin(q[0])*np.sin(q[1])*np.sin(q[3]) + 3*np.sin(q[0])*np.cos(q[1])*np.cos(q[2])
                    ,-2*np.cos(q[3])*np.cos(q[0])*np.cos(q[1])*np.cos(q[2]) + 2*np.cos(q[0])*np.sin(q[1])*np.sin(q[3]) - 3*np.cos(q[0])*np.cos(q[1])*np.cos(q[2])
                    ,-2*np.sin(q[1])*np.cos(q[2])*np.cos(q[3]) - 2*np.cos(q[1])*np.sin(q[3]) - 3*np.cos(q[2])*np.sin(q[1])])
        J3 = np.array([2*np.cos(q[3])*(-np.sin(q[0])*np.sin(q[1])*np.sin(q[2]) + np.cos(q[2])*np.cos(q[1])) - 3*np.sin(q[0])*np.sin(q[1])*np.sin(q[2]) + 3*np.cos(q[2])*np.cos(q[0])
                    ,2*np.cos(q[3])*(np.cos(q[0])*np.sin(q[1])*np.sin(q[2]) + np.sin(q[0])*np.cos(q[2])) + 3*(np.cos(q[0])*np.sin(q[1])*np.sin(q[2]) + np.cos(q[2])*np.sin(q[0]))
                    ,-2*np.cos(q[1])*np.sin(q[2])*np.cos(q[3]) - 3*np.sin(q[2])*np.cos(q[1])])
        J4 = np.array([-2*np.sin(q[3])*(np.sin(q[0])*np.sin(q[1])*np.cos(q[2]) + np.sin(q[2])*np.cos(q[0])) + 2*np.sin(q[0])*np.cos(q[1])*np.cos(q[3])
                    ,2*np.sin(q[3])*(np.cos(q[0])*np.sin(q[1])*np.cos(q[2]) - np.sin(q[0])*np.sin(q[2])) - 2*np.cos(q[0])*np.cos(q[1])*np.cos(q[3])
                    ,-2*(np.cos(q[1])*np.cos(q[2])*np.sin(q[3]) + np.sin(q[1])*np.cos(q[3]))])
        Jacobian = np.array([J1,J2,J3,J4]).transpose()  #transpose because J1 to J4 are initially lines, we want them as columns
        return Jacobian

      # Estimate control inputs for closed-loop control, with q input angles, pos_d the desired position and pos_c the current position
    def closed_loop_control(self,q,pos_c,pos_d):
        # PI parameters
        Kp = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]])
        Kd = np.array([[0.005, 0, 0], [0, 0.005, 0], [0, 0, 0.005]])
        # estimate time step
        cur_time = np.array([rospy.get_time()])
        dt = cur_time - self.time_previous_step
        self.time_previous_step = cur_time

        self.d_error = ((pos_d - pos_c) - self.error)/dt
        self.error = pos_d-pos_c
        # calculating an approximation of the psudeo inverse of Jacobian
        J_inv = self.calculate_jacobian(q).transpose()
        dq_d = np.dot(J_inv, (np.dot(Kp,self.error.transpose()) ) )
        q_d = (q + (dt * dq_d))  # control input (angular position of joints)

        return q_d

    # Calculates the 3D positions of the different blobs we want to know about.
    def calculate_blob_positions(self, img1, img2):
        # Absolute value of the location of the yellow sphere
        yellow_sphere_location = np.array([0, 0, 0.5])
        # Output array
        master_positions = np.zeros(shape=(5,3))

        positions1 = self.get_positions(img1)
        positions2 = self.get_positions(img2)

        # Process image 1's determined positions.
        for i, pos in enumerate(positions1):
            master_positions[i] = np.array([-1, pos[0], pos[1]])

            self.draw_spot_at(img1, pos)

        # Process image 2's determined positions. (Using the average Z position)
        for i, pos in enumerate(positions2):
            old_pos = master_positions[i].copy()

            avg = (old_pos[2] + pos[1])/2 if old_pos[2] != -1 and pos[1] != -1 else -1

            master_positions[i] = np.array([pos[0], old_pos[1], avg])

            self.draw_spot_at(img2, pos)

        # If we don't happen to have any positional data for one of the blobs, use the last known position.
        for i in range(5):
            for j in range(3):
                if master_positions[i][j] == -1:
                    master_positions[i][j] = self.last_known_locations[i][j]
                else:
                    self.last_known_locations[i][j] = master_positions[i][j]

        world_center = master_positions[0].copy()
        for i in range(5):
            master_positions[i] -= world_center

        # Convert pixel distances into real world distances
        # Value calculated manually on an earlier run of the program
        master_positions /= 25.6

        for i in range(5):
            # Since the y coordinate of the images is flipped, we need to flip it again to
            # get back to sensible real world results.
            master_positions[i][2] *= -1
            # Add the yellow sphere location to the results to get them into world space.
            master_positions[i] += yellow_sphere_location

        return master_positions


    # Calculates the angle between three 3D points.
    def angle_between_three_points(self, a, b, c):
        ba = a - b
        bc = c - b
        return self.angle_between_two_vectors(ba, bc)

    # Calculates the angle between two 3D vectors.
    def angle_between_two_vectors(self, ba, bc):
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return np.arccos(cosine_angle)

    def magnitude(self, x, y):
        return math.sqrt(x*x+y*y)

    # Establishes the desired angles of joints 1, 2 and 3 based on the provided angle and
    # the 'target' position (The position of the green blob relative to the yellow blob)
    def calculate_joints_1_2_3(self, green_blob_vector, angle):
        green_blob_vector = list(green_blob_vector)

        # Use the equation found on https://en.wikipedia.org/wiki/Universal_joint to calculate the angle to rotate joint1 by.
        magnitude_of_input = self.magnitude(green_blob_vector[0], green_blob_vector[1])
        angle_of_input = math.atan(green_blob_vector[2] / magnitude_of_input)
        beta = math.pi / 2 - angle_of_input
        clamped_angle = angle % (math.pi * 2)
        angle = math.atan(math.cos(beta) * math.tan(angle))

        # Modify the angle so it's in a state useful for calculating joint 1
        angle += math.pi if math.pi / 2 < clamped_angle < math.pi * 3 / 2 else 0
        angle = (angle + math.pi) % (math.pi * 2) - math.pi

        # Calculate joint 1's angle. It is simply a sum of the desired universal joint angle and
        # the initial angle of the input vector.
        joint1 = (-angle + math.atan2(green_blob_vector[1], green_blob_vector[0])) % (math.pi * 2) - math.pi

        # Modify the angle again so it's more useful for calculating the target and transformed_target angles
        angle = (angle if angle < 0 else math.pi - angle) + math.pi * 3 / 4

        # Establish a new working vector, rotated so we're now working in joint 1's transform.
        c = math.cos(angle)
        s = math.sin(angle)
        new_vector = [c - s, s + c, green_blob_vector[2]]

        # Calculate the angle of joint2 so it's pointing in the direction of the new vector.
        joint2 = -math.atan(new_vector[1] / new_vector[2])
        # Calculate the angle of joint3 in a similar way.
        joint3 = math.atan(new_vector[0] / self.magnitude(new_vector[1], new_vector[2]) * math.copysign(1, new_vector[1]))

        return joint1, joint2, joint3

    def calculate_universal_joint_rotation(self, master_positions):
        # Calculate the basic angle of the red joint relative to the green and yellow ones by finding intersecting plane angles.
        normal_1 = np.cross(master_positions[0] - master_positions[1], master_positions[1] - master_positions[2])
        normal_2 = np.cross(master_positions[1] - master_positions[2], master_positions[2] - master_positions[3])
        angle = (self.angle_between_two_vectors(normal_1, normal_2))
        angle -= math.pi / 2

        # Establish whether the plane is intersecting 'backwards' or not - The angle between two planes
        # is not signed, but we want our output to be signed.
        d = np.dot(normal_1, master_positions[2])
        c = np.dot(normal_1, master_positions[3])
        backward = math.copysign(1, c - d)

        # Alter the angle based on whether we're intersecting backwards or not.
        old_angle = angle
        if backward == 1 and old_angle < 0:
            angle *= -1
        if backward == -1 and old_angle < 0:
            angle += math.pi
        if backward == -1 and old_angle > 0:
            angle += math.pi
        if backward == 1 and old_angle > 0:
            angle = math.pi * 2 - angle

        return angle

    def calculate_joint_angles(self, master_positions):
        angle1 = self.calculate_universal_joint_rotation(master_positions)
        # The direction we want the green blob to point in
        direction = master_positions[2] - master_positions[1]
        joint1, joint2, joint3 = self.calculate_joints_1_2_3(direction, angle1)
        joint4 = math.pi - self.angle_between_three_points(master_positions[1], master_positions[2], master_positions[3])

        # Rotate all of the angles to ensure that they're between 0 and pi/2
        if joint1 < 0:
            joint1 += math.pi
            joint2 *= -1
            joint3 *= -1
            joint4 *= -1
        if joint1 > math.pi/2:
            joint1 = math.pi-joint1
            joint3 *= -1
            joint4 *= -1

        return (joint1, joint2, joint3, joint4)

    # Receive data and save it for camera 1's callback.
    def callback1(self, data):
        self.cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8").copy()


    # Receive data from camera 1, process it, and publish
    def callback2(self, data):
        #self.estimating = (time.time() % 5 < 2.5)
        self.estimating = False

        self.cv_image2 = self.bridge.imgmsg_to_cv2(data, "bgr8").copy()

        size = (400, 400)
        centre1 = (400, 500)
        centre2 = (400, 500)

        # Crop the images so we're not doing unnecessary work
        img1 = self.cv_image1[
               centre1[1] - size[1] / 2:centre1[1] + size[1] / 2,
               centre1[0] - size[0] / 2:centre1[0] + size[0] / 2
               ].copy()
        img2 = self.cv_image2[
               centre2[1] - size[1] / 2:centre2[1] + size[1] / 2,
               centre2[0] - size[0] / 2:centre2[0] + size[0] / 2
               ].copy()

        master_positions = self.calculate_blob_positions(img1, img2)

        (joint1, joint2, joint3, joint4) = self.calculate_joint_angles(master_positions)

        #if self.estimating: # Remove not
            #self.my_estimation = (joint1, joint2, joint3, joint4)

        self.my_estimation = (joint1, joint2, joint3, joint4)
        cv2.imshow('window1', np.asarray(img1))
        cv2.imshow('window2', np.asarray(img2))

        cv2.waitKey(1)

        # current set of angles
        q = np.array(self.q_d)
        # Current position of the end_effector
        end_effector_pos = self.ForwardK(q)
        # position of the target
        target_p = master_positions[4]

        self.q_d = self.closed_loop_control(q,end_effector_pos, target_p)

        # Publish the results
        try:
            self.image_pub1.publish(self.bridge.cv2_to_imgmsg(img1, "bgr8"))
            self.image_pub2.publish(self.bridge.cv2_to_imgmsg(img2, "bgr8"))

            #publish the estimated position of the target
            self.target_x_pub.publish(master_positions[4][0])
            self.target_y_pub.publish(master_positions[4][1])
            self.target_z_pub.publish(master_positions[4][2])

            testing_joint_estimation = False # Change this to true to test the joint estimation code out.
            if testing_joint_estimation:
                #if self.estimating:
                    #joint1, joint2, joint3, joint4 = (0.9, -0.3, -0.2, -1)
                #else:
                    #joint1, joint2, joint3, joint4 = self.my_estimation

                joint1, joint2, joint3, joint4 = self.my_estimation
                print(joint1, joint2, joint3, joint4)

                #self.robot_joint1_pub.publish(joint1)
                #self.robot_joint2_pub.publish(joint2)
                #self.robot_joint3_pub.publish(joint3)
                #self.robot_joint4_pub.publish(joint4)
            else:
                #send the desired angles to the robot so that it can follow the target
                self.robot_joint1_pub.publish(self.q_d[0])
                self.robot_joint2_pub.publish(self.q_d[1])
                self.robot_joint3_pub.publish(self.q_d[2])
                self.robot_joint4_pub.publish(self.q_d[3])

            print ("Target measured :", master_positions[4])
            print ("end effector pos",  end_effector_pos)
            print ("\n")

            DD = math.sqrt(((master_positions[4][0]-end_effector_pos[0])**2)+((master_positions[4][1]-end_effector_pos[1])**2)+((master_positions[4][2]-end_effector_pos[2])**2))
            self.DD_pub.publish(DD)

            #publish the results of the FK equations in a topic to plot it afterwards
            self.end_effector_x_FK.publish(end_effector_pos[0])
            self.end_effector_y_FK.publish(end_effector_pos[1])
            self.end_effector_z_FK.publish(end_effector_pos[2])

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
