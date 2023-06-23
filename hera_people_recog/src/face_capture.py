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
    recog = 0

    def __init__(self):
        rospy.Service('face_captures', face_capture, self.handler)
        rospy.loginfo("Start FaceRecogniser Init process...")

        self.rate = rospy.Rate(5)
        rospack = rospkg.RosPack()
        self.path_to_package = rospack.get_path('hera_face')

        self.bridge_object = CvBridge()
        self.topic = "/zed_node/left_raw/image_raw_color"
        self._check_cam_ready()
        self.image_sub = rospy.Subscriber(self.topic, Image, self.camera_callback)

        self._bridge = CvBridge()

        rospy.loginfo("Finished FaceRecogniser Init process, ready to capture")

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

    def capture(self, data, request):
        try:
            video_capture = self.bridge_object.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)

        small_frame = cv2.resize(video_capture, (0, 0), fx=1, fy=1)

        image_path = os.path.join(self.path_to_package + '/face_images/')
        detector = dlib.get_frontal_face_detector()

        face_locations = detector(small_frame, 1)

        if len(face_locations) <= 0:
            rospy.logwarn("No Faces found, please get closers...")
        else:
            write_status = cv2.imwrite(str(image_path) + request.name + '.jpg', small_frame)

            if write_status is True:
                rospy.loginfo('Face ' + request.name + ' saved succeeded!')

                rospy.loginfo(str(image_path) + request.name + '.jpg')
                self.recog = 1
                return True
            else:
                rospy.loginfo('Face not saved!')
                return False

    def handler(self, request):
        self.recog = 0

        while self.recog == 0:
            resp = self.capture(self.cam_image, request)
            self.rate.sleep()
            if self.recog == 1:
                return resp

        cv2.destroyAllWindows()


if __name__ == '__main__':
    rospy.init_node('face_captures', log_level=rospy.INFO)
    FaceCapture()

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
