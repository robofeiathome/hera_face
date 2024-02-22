#!/usr/bin/env python3
import os
import rospy
from sensor_msgs.msg import Image
import rospkg
from hera_face.srv import face_list
from cv_bridge import CvBridge
import dlib
import numpy as np
import fnmatch

class FaceRecog:
    def __init__(self):
        rospy.Service('face_recog', face_list, self.handler)
        self.rate = rospy.Rate(5)

        rospack = rospkg.RosPack()
        self.topic = rospy.get_param('~camera_topic')
        self.bridge = CvBridge()
        self.path_to_package = rospack.get_path('hera_face')
        self.init_model()

        self.cam_image = None
        self.image_sub = rospy.Subscriber(self.topic, Image, self.camera_callback)

        rospy.loginfo("Finished FaceRecog Init process, ready to recognise")

    def init_model(self):
        sp_path = rospy.get_param('~sp_path')
        model_path = rospy.get_param('~model_path')
        db_path = rospy.get_param('~db_path')
        self.detector = dlib.get_frontal_face_detector()
        self.sp = dlib.shape_predictor(f'{self.path_to_package}{sp_path}')
        self.model = dlib.face_recognition_model_v1(f'{self.path_to_package}{model_path}')
        self.people_dir = f'{self.path_to_package}/{db_path}'
        self.load_data()

    def camera_callback(self, data):
        self.cam_image = data

    def load_data(self):
        self.known_faces = []
        self.known_names = []
        for file_name in fnmatch.filter(os.listdir(self.people_dir), '*.jpg'):
            path = os.path.join(self.people_dir, file_name)
            img = dlib.load_rgb_image(path)
            detections = self.detector(img, 1)
            if detections:
                shape = self.sp(img, detections[0])
                face_descriptor = self.model.compute_face_descriptor(img, shape)
                self.known_faces.append(np.array(face_descriptor))
                self.known_names.append(file_name[:-4].lower())

    def recognize(self, img):
        detections = self.detector(img, 1)
        faces_encodings = []
        centers = []

        for detection in detections:
            face_encoding = self.model.compute_face_descriptor(img, self.sp(img, detection))
            faces_encodings.append(face_encoding)

            center_x = (detection.left() + detection.right()) / 2
            centers.append(center_x)

        return len(detections), faces_encodings, centers


    def find_matches(self, faces_encodings):
        names = []
        for encoding in faces_encodings:
            matches = [name for name, known_encoding in zip(self.known_names, self.known_faces) 
                       if np.linalg.norm(known_encoding - encoding) <= 0.6]
            names.append(matches[0] if matches else "unknown")
        return names

    def handler(self, request):
        if self.cam_image is None:
            return ['no image'], [0.0], 0

        cv_image = self.bridge.imgmsg_to_cv2(self.cam_image, "bgr8")
        num_faces, faces_encodings, centers = self.recognize(cv_image)
        names = self.find_matches(faces_encodings)

        if request.name != '': 
            if request.name in names:
                index = names.index(request.name)
                return [request.name], [centers[index]], num_faces
            else:
                return [], [], num_faces
        else:
            return names, centers, num_faces


if __name__ == '__main__':
    rospy.init_node('face_recog', anonymous=True)
    fr = FaceRecog()
    rospy.spin()
