#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from hera_face.srv import face_check, face_checkResponse
from hera_control.srv import Joint_service, Manip_service
import cv2
import dlib
from cv_bridge import CvBridge, CvBridgeError
import time

class FaceCheck:
    def __init__(self):
        self.bridge_object = CvBridge()
        self.topic = rospy.get_param('~camera_topic')
        self.image_sub = rospy.Subscriber(self.topic, Image, self.camera_callback)
        self.service = rospy.Service('face_check', face_check, self.handler)

        self.manipulator_service = rospy.ServiceProxy('/manipulator', Manip_service)
        self.joint_command = rospy.ServiceProxy('/joint_command', Joint_service)
        rospy.loginfo("Finished FaceCheck Init process, ready to check")

    def camera_callback(self, data):
        """Callback function that receives camera images."""
        self.cam_image = data

    def face_check(self):
        """Checks if there are any faces in the given image."""
        try:
            cv_image = self.bridge_object.imgmsg_to_cv2(self.cam_image, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return False

        # No need to resize if we are processing with 1:1 scale
        small_frame = cv2.resize(cv_image, (0, 0), fx=1, fy=1)
        face_detector = dlib.get_frontal_face_detector()
        face_locations = face_detector(small_frame, 1)

        if face_locations:
            rospy.loginfo("Face Found!")
            return True
        else: 
            rospy.loginfo("No face found!")
            return False

    def head_goal(self, type):
        """Move the head of the robot to the desired position."""
        self.manipulator_service(type=type)
        
    def head_command(self, rad):
        """Move the head of the robot to the desired position."""
        self.joint_command(10, rad)

    def check_with_move(self):
        """Check if there are faces in the image and move the head if not."""
        HEAD_TILT = ['way_up', 'head_up', 'head_down']
        i = 0
        
        while not self.face_check():
            try: 
                self.head_goal(HEAD_TILT[i])
                i = i + 1 if i < 2 else 0
                time.sleep(1)
            except rospy.ServiceException as e:
                rospy.logwarn("Joint Command Service call failed: %s" % e)
                return False
        return True

    def handler(self, request):
        """Service handler to perform face check."""
        if self.cam_image is None:
            rospy.logwarn("No image available from camera.")
            return False

        if request.move_head:
            return self.check_with_move()

        return self.face_check()



if __name__ == '__main__':
    rospy.init_node('face_check', log_level=rospy.INFO)
    face_checker = FaceCheck()

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
