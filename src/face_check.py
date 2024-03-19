#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from hera_face.srv import face_check, face_checkResponse
import cv2
import dlib
from cv_bridge import CvBridge, CvBridgeError

class FaceCheck:
    def __init__(self):
        self.bridge_object = CvBridge()
        self.topic = rospy.get_param('~camera_topic')
        self.image_sub = rospy.Subscriber(self.topic, Image, self.camera_callback)
        self.service = rospy.Service('face_check', face_check, self.handler)
        rospy.loginfo("Finished FaceCheck Init process, ready to check")

    def camera_callback(self, data):
        """Callback function that receives camera images."""
        self.cam_image = data

    def face_check(self, img):
        """Checks if there are any faces in the given image."""
        try:
            cv_image = self.bridge_object.imgmsg_to_cv2(img, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return False

        # No need to resize if we are processing with 1:1 scale
        small_frame = cv2.resize(cv_image, (0, 0), fx=1, fy=1)
        face_detector = dlib.get_frontal_face_detector()
        face_locations = face_detector(small_frame, 1)

        if not face_locations:
            rospy.logwarn("No Faces found, please get closer...")
            return False

        rospy.loginfo("Face found, welcome!")
        return True

    def handler(self, request):
        """Service handler to perform face check."""
        if self.cam_image is None:
            rospy.logwarn("No image available from camera.")
            return False

        result = self.face_check(self.cam_image)
        return result

if __name__ == '__main__':
    rospy.init_node('face_check', log_level=rospy.INFO)
    face_checker = FaceCheck()

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
