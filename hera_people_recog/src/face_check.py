#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from hera_face.srv import face_check
import cv2
import dlib
from cv_bridge import CvBridge, CvBridgeError


class FaceCheck:
    def __init__(self):
        self.bridge_object = CvBridge()
        self.topic = "/zed_node/left_raw/image_raw_color"
        self.check_cam_ready()
        self.image_sub = rospy.Subscriber(self.topic, Image, self.camera_callback)
        self.service = rospy.Service('face_check', face_check, self.handler)
        self.rate = rospy.Rate(3)
        rospy.loginfo("Finished FaceCheck Init process, ready to check")

    def check_cam_ready(self):
        self.cam_image = None
        while self.cam_image is None and not rospy.is_shutdown():
            try:
                self.cam_image = rospy.wait_for_message(self.topic, Image, timeout=1.0)
                rospy.logdebug(f"Current {self.topic} READY=>{str(self.cam_image)}")
            except:
                rospy.logerr(f"Current {self.topic} not ready yet, retrying.")

    def camera_callback(self, data):
        self.cam_image = data

    def face_check(self, data):
        try:
            video_capture = self.bridge_object.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)

        small_frame = cv2.resize(video_capture, (0, 0), fx=1, fy=1)
        face_locations = dlib.get_frontal_face_detector()(small_frame, 1)

        if not face_locations:
            rospy.logwarn("No Faces found, please get closer...")
            return False

        rospy.loginfo("Face found, welcome!")
        return True

    def handler(self, request):
        while True:
            if self.face_check(self.cam_image):
                return True
            self.rate.sleep()

        cv2.destroyAllWindows()


if __name__ == '__main__':
    rospy.init_node('face_check', log_level=rospy.INFO)
    face_checker = FaceCheck()

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
