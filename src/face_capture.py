#!/usr/bin/env python3
import os
import rospy
import rospkg
from sensor_msgs.msg import Image
from hera_face.srv import face_capture
import cv2
import dlib
from cv_bridge import CvBridge, CvBridgeError

class FaceCapture:
    def __init__(self):
        self.bridge_object = CvBridge()
        self.topic = rospy.get_param('~camera_topic')
        self.image_sub = rospy.Subscriber(self.topic, Image, self.camera_callback)
        self.service = rospy.Service('face_captures', face_capture, self.handler)
        self.rospack = rospkg.RosPack()
        self.path_to_package = self.rospack.get_path('hera_face')
        rospy.loginfo("Finished FaceCapture Init process, ready to capture")

    def camera_callback(self, data):
        """Stores the received camera image."""
        self.cam_image = data

    def capture(self, data, request):
        """Attempts to capture a face from the provided image data."""
        try:
            cv_image = self.bridge_object.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return False

        small_frame = cv2.resize(cv_image, (0, 0), fx=1, fy=1)
        face_detector = dlib.get_frontal_face_detector()
        face_locations = face_detector(small_frame, 1)

        if not face_locations:
            rospy.logwarn("No Faces found, please get closer...")
            return False

        # Assumes saving the entire frame if a face is detected
        image_path = os.path.join(self.path_to_package, 'face_images', f"{request.name}.jpg")
        cv2.imwrite(image_path, small_frame)
        rospy.loginfo(f'Face {request.name} saved at {image_path}')
        return True

    def handler(self, request):
        """Handles face capture requests."""
        if not self.cam_image:
            rospy.logwarn("No image available.")
            return False

        result = self.capture(self.cam_image, request)
        return result

if __name__ == '__main__':
    rospy.init_node('face_captures', anonymous=True)
    face_capturer = FaceCapture()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
