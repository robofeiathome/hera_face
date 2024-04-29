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
import cv2

class FaceRecog:
    def __init__(self):
        rospy.Service('face_recog', face_list, self.handler)
        self.rate = rospy.Rate(5)

        rospack = rospkg.RosPack()
        self.topic = rospy.get_param('~camera_topic')
        self.log_path = rospy.get_param('~log_path')
        self.bridge = CvBridge()
        self.path_to_package = rospack.get_path('hera_face')
        self.init_model()
        self.erase_log()

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

    def erase_log(self):
        for file_name in os.listdir(f'{self.path_to_package}/{self.log_path}'):
            file_path = os.path.join(f'{self.path_to_package}/{self.log_path}', file_name)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)

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

    def recognise(self, img):
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
            names.append(matches[0] if matches else "face")
        return names
     
    def draw_bounding_boxes(self, img, detections, names):
        for det, name in zip(detections, names):
            left, top, right, bottom = det.left(), det.top(), det.right(), det.bottom()
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(img, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def recognise_and_save(self, img):
        num_faces, faces_encodings, centers = self.recognise(img)
        names = self.find_matches(faces_encodings)

        self.draw_bounding_boxes(img, self.detector(img, 1), names)

        save_path = f'{self.path_to_package}/{self.log_path}/face_recog_{rospy.Time.now()}.jpg'
        cv2.imwrite(save_path, img)
        rospy.loginfo(f"Image saved to {save_path}")

        return num_faces, names, centers

    def handler(self, request):
        if self.cam_image is None:
            return ['no image'], [0.0], 0
        print("request: ", request.name)
        cv_image = self.bridge.imgmsg_to_cv2(self.cam_image, "bgr8")
        self.load_data()
        num_faces, names, centers = self.recognise_and_save(cv_image)
        print("names: ", names)
        print("centers: ", centers)
        if request.name != '': 
            request_name = request.name.lower()
            if request_name in names:
                index = names.index(request_name)
                print("I will return the name in the request: ", request_name)
                return [request.name], [centers[index]], num_faces
            else:
                print("I will return an empty list, I did not find the name in the request.", request.name)
                return [], [], num_faces
        else:
            print("I will return the names and centers found in the image, was not indentified.")
            return names, centers, num_faces


if __name__ == '__main__':
    rospy.init_node('face_recog', anonymous=True)
    fr = FaceRecog()
    rospy.spin()
