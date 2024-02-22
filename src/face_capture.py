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
        self.topic = "/zed_node/left_raw/image_raw_color"
        self.check_cam_ready()
        self.image_sub = rospy.Subscriber(self.topic, Image, self.camera_callback)
        self.service = rospy.Service('face_captures', face_capture, self.handler)
        self.rate = rospy.Rate(5)
        self.rospack = rospkg.RosPack()
        self.path_to_package = self.rospack.get_path('hera_face')
        rospy.loginfo("Finished FaceRecogniser Init process, ready to capture")

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

    def capture(self, data, request):
        try:
            video_capture = self.bridge_object.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)

        small_frame = cv2.resize(video_capture, (0, 0), fx=1, fy=1)
        image_path = os.path.join(self.path_to_package, 'face_images')
        face_locations = dlib.get_frontal_face_detector()(small_frame, 1)

        if not face_locations:
            rospy.logwarn("No Faces found, please get closer...")
            return False

        image_file_path = os.path.join(image_path, f"{request.name}.jpg")
        write_status = cv2.imwrite(image_file_path, small_frame)

        if write_status:
            rospy.loginfo(f'Face {request.name} saved succeeded!')
            rospy.loginfo(image_file_path)
            return True

        rospy.loginfo('Face not saved!')
        return False

    def handler(self, request):
        while True:
            if self.capture(self.cam_image, request):
                return True
            self.rate.sleep()

        cv2.destroyAllWindows()


if __name__ == '__main__':
    rospy.init_node('face_captures', log_level=rospy.INFO)
    face_capturer = FaceCapture()

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
