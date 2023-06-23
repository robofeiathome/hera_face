#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from hera_face.srv import face_check

import cv2
import dlib
from cv_bridge import CvBridge, CvBridgeError

class FaceCheck:
    detect = 0

    def __init__(self):
        rospy.Service('face_check', face_check, self.handler)

        rospy.loginfo("Start FaceRecogniser Init process...")
        self.rate = rospy.Rate(5)

        self.bridge_object = CvBridge()
        rospy.loginfo("Start camera suscriber...")
        self.topic = "/zed_node/left_raw/image_raw_color"
        self._check_cam_ready()
        self.image_sub = rospy.Subscriber(self.topic, Image, self.camera_callback)

        self._bridge = CvBridge()

        rospy.loginfo("Finished FaceCheck Init process, ready to check")

    def _check_cam_ready(self):
        self.cam_image = None
        while self.cam_image is None and not rospy.is_shutdown():
            try:
                self.cam_image = rospy.wait_for_message(self.topic, Image, timeout=1.0)
                rospy.logdebug("Current " + self.topic + " READY=>" + str(self.cam_image))
            except:
                rospy.logerr("Current " + self.topic + " not ready yet, retrying.")

    def camera_callback(self, data):
        self.cam_image = data

    def face_check(self, data):
        try:
            video_capture = self.bridge_object.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)

        small_frame = cv2.resize(video_capture, (0, 0), fx=0.5, fy=0.5)

        detector = dlib.get_frontal_face_detector()
        face_locations = detector(small_frame, 1)
        print(face_locations)

        if len(face_locations) <= 0:
            rospy.logwarn("No Faces found, please get closer...")
            return False

        else:
            rospy.loginfo("Face found, welcome!")
            self.detect = 1
            return True

    def handler(self, request):
        self.detect = 0

        while self.detect == 0:
            resp = self.face_check(self.cam_image)
            self.rate.sleep()
            return resp

        cv2.destroyAllWindows()


if __name__ == '__main__':
    rospy.init_node('face_check', log_level=rospy.INFO)
    FaceCheck()

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
