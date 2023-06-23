#!/usr/bin/env python3
import os
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
import rospkg
from hera_face.srv import face_list

from cv_bridge import CvBridge
from ultralytics import YOLO
import dlib
import numpy as np
import fnmatch
import cv2
import time


class FaceRecog:
    recog = 0

    def __init__(self):
        rospy.Service('face_recog', face_list, self.handler)
        rospy.loginfo("Start FaceRecogniser Init process...")
        self.rate = rospy.Rate(5)
        rospack = rospkg.RosPack()

        self.path_to_package = rospack.get_path('hera_face')
        self.yolo = YOLO(self.path_to_package + '/src/coco.pt')
        self.bridge_object = CvBridge()
        rospy.loginfo("Start camera suscriber...")
        self.topic = "/zed_node/left_raw/image_raw_color"
        self._check_cam_ready()
        self.image_sub = rospy.Subscriber(self.topic, Image, self._camera_callback)

        self.center_place = None
        self.detector = dlib.get_frontal_face_detector()
        self.sp = dlib.shape_predictor(os.path.join(self.path_to_package, "src/shape_predictor_5_face_landmarks.dat"))
        self.model = dlib.face_recognition_model_v1(
            os.path.join(self.path_to_package, "src/dlib_face_recognition_resnet_model_v1.dat"))
        self.people_dir = os.path.join(self.path_to_package, 'face_images')
        self.face_center = []
        self.face_name = []
        self.known_face = []
        self.known_name = []

        self.pub_cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        rospy.loginfo("Finished FaceRecogniser Init process...Ready")

    def _check_cam_ready(self):
        self.cam_image = None
        while self.cam_image is None and not rospy.is_shutdown():
            try:
                self.cam_image = rospy.wait_for_message(self.topic, Image, timeout=1.0)
                rospy.logdebug("Current " + self.topic + " READY=>" + str(self.cam_image))
            except:
                rospy.logerr("Current " + self.topic + " not ready yet, retrying.")

    def _camera_callback(self, data):
        self.cam_image = data

    def _load_data(self):
        files = fnmatch.filter(os.listdir(self.people_dir), '*.jpg')

        for file_name in files:
            img = dlib.load_rgb_image(os.path.join(self.people_dir, file_name))
            img_detected = self.detector(img, 1)
            bounding_boxes = []

            if len(img_detected) > 0:
                for i, rects in enumerate(img_detected):
                    area = rects.top() * rects.left()
                    bounding_boxes.append(area)
                biggest_box = bounding_boxes.index(max(bounding_boxes))
                img_shape = self.sp(img, img_detected[biggest_box])
                align_img = dlib.get_face_chip(img, img_shape)
                img_rep = np.array(self.model.compute_face_descriptor(align_img))
                self.known_face.append(img_rep)
                self.known_name.append(file_name.split('.')[0])
            else:
                rospy.loginfo("No face detected in image: " + file_name)

    def _calculate_media(self, x1, x2):
        return (x1 + x2) / 2

    def _spin(self, velocidade=-0.4, sleep_time=0.0):
        vel_cmd = Twist()
        vel_cmd.angular.z = velocidade
        self.pub_cmd_vel.publish(vel_cmd)

    def _predict(self):
        small_frame = self.bridge_object.imgmsg_to_cv2(self.cam_image, desired_encoding="bgr8")
        results = self.yolo.predict(source=small_frame, conf=0.3, device=0, classes=[56, 57])
        print('Len boxes: ', len(results[0].boxes))
        return results[0].boxes

    def _update_boxes(self):
        return self._predict()

    def _find_sit(self):
        while True:
            print('single loop')
            boxes = self._predict()
            if len(boxes) > 0:
                print('box > 0')
                self._spin(0, 1)
                boxes = self._update_boxes()
                self.center_place = self._find_empty_place(boxes)

            if self.center_place is not None:
                print('break')
                break
            else:
                print('_spin and detect')
                self._spin(-0.4, 2)

        return self.center_place

    def _find_empty_place(self, boxes):
        center_place = None
        print('Looking for an empty place')
        for k, c in enumerate(boxes.cls):
            box = boxes[k].xyxy[0]
            obj_class = self.yolo.names[int(c)]
            print(obj_class)

            found = False
            if self.face_center:
                for i, face_name in enumerate(self.face_name):
                    if face_name in self.known_name:
                        found = True
                        print('box0:', box[0])
                        print('box2:', box[2])
                        print('center:', self.face_center[i])

                        if obj_class == 'chair' and not any(box[0] < center_face < box[2] for center_face in self.face_center):
                            center_place = self._calculate_media(box[0], box[2])
                            print("lugar 0")
                        elif obj_class == 'couch':
                            media_x = self._calculate_media(box[0], box[2])
                            if not any(box[0] < center_face < media_x for center_face in self.face_center):
                                center_place = self._calculate_media(box[0], media_x)
                                print('lugar 1')
                            elif not any(media_x < center_face < box[2] for center_face in self.face_center):
                                center_place = self._calculate_media(media_x, box[2])
                                print('lugar 2')

                if not found:
                    print('not Found condition')
                    if obj_class == 'chair':
                        center_place = self._calculate_media(box[0], box[2])
                    elif obj_class == 'couch':
                        media_x = (box[0] + box[2]) / 2
                        center_place = self._calculate_media(box[0], media_x)
            else:
                print('Do not recognized face')
                if obj_class == 'chair':
                    center_place = self._calculate_media(box[0], box[2])
                elif obj_class == 'couch':
                    media_x = self._calculate_media(box[0], box[2])
                    center_place = self._calculate_media(box[0], media_x)

        return center_place

    def _recognize(self, small_frame):
        img_detected = self.detector(small_frame, 1)

        if len(img_detected) == 0:
            rospy.loginfo("No face detected")
            self._find_sit()
            return len(img_detected)

        faces = dlib.full_object_detections()
        for detection in img_detected:
            faces.append(self.sp(small_frame, detection))
        align_img = dlib.get_face_chips(small_frame, faces)
        img_rep = np.array(self.model.compute_face_descriptor(align_img))

        for i in range(len(img_detected)):
            name = 'Face'
            for j in range(len(self.known_face)):
                euclidean_dist = list(np.linalg.norm(self.known_face - img_rep[i], axis=1) <= 0.62)
                if True in euclidean_dist:
                    fst = euclidean_dist.index(True)
                    name = self.known_name[fst]
                else:
                    continue
            self.face_name.insert(i, name)

        for i, rects in enumerate(img_detected):
            if self.face_name[i] in self.known_name:
                cv2.rectangle(small_frame, (rects.left(), rects.top()), (rects.right(), rects.bottom()), (0, 255, 0), 2)
                cv2.putText(small_frame, self.face_name[i], (rects.left(), rects.top()), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.rectangle(small_frame, (rects.left(), rects.top()), (rects.right(), rects.bottom()), (255, 0, 0), 2)
                cv2.putText(small_frame, self.face_name[i], (rects.left(), rects.top()), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 0, 0), 2, cv2.LINE_AA)
            center_x = (rects.right() + rects.left()) / 2
            self.face_center.append(center_x)

        window = dlib.image_window()
        window.set_image(small_frame)
        cv2.imwrite(self.path_to_package + '/face_recogs/recog.jpg', small_frame)

        return len(img_detected)

    def start(self, data, nome_main):
        self._load_data()

        small_frame = self.bridge_object.imgmsg_to_cv2(data, desired_encoding="bgr8")

        num_faces = self._recognize(small_frame)

        name = ''
        center = 0.0
        if nome_main == '':
            self._find_sit()
            print(self.center_place)
            self.recog = 1
        elif nome_main in self.face_name:
            name = nome_main
            center = self.face_center[self.face_name.index(nome_main)]
            self.recog = 1
        else:
            name = 'face'
            center = 0.0
            self.recog = 1
        return name, center, num_faces, self.center_place

    def handler(self, request):
        self.recog = 0
        while self.recog == 0:
            self.image_sub = rospy.Subscriber(self.topic, Image, self._camera_callback)
            name, center, num, empty = self.start(self.cam_image, request.name)
            self.rate.sleep()
            return name, float(center), num, empty
        cv2.destroyAllWindows()


if __name__ == '__main__':
    rospy.init_node('face_recog', log_level=rospy.INFO)
    FaceRecog()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
