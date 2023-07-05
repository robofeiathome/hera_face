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
        """
        Constructor for the FaceRecog class.
        Initializes necessary ROS parameters, services and publishers.
        Also, it sets up the face recognition model and other necessary parameters.
        """
        rospy.Service('face_recog', face_list, self.handler)
        rospy.loginfo("Start FaceRecogniser Init process...")
        self.rate = rospy.Rate(5)
        rospack = rospkg.RosPack()

        self.path_to_package = rospack.get_path('hera_face')
        self.yolo = YOLO(self.path_to_package + '/src/coco.pt')
        self.bridge_object = CvBridge()
        rospy.loginfo("Start camera suscriber...")
        self.topic = "/zed_node/right_raw/image_raw_color"
        self._check_cam_ready()
        self.image_sub = rospy.Subscriber(self.topic, Image, self.camera_callback)

        self.center_place = None
        self.detector = dlib.get_frontal_face_detector()
        self.sp = dlib.shape_predictor(os.path.join(self.path_to_package, "src/shape_predictor_5_face_landmarks.dat"))
        self.model = dlib.face_recognition_model_v1(
            os.path.join(self.path_to_package, "src/dlib_face_recognition_resnet_model_v1.dat"))
        self.people_dir = os.path.join(self.path_to_package, 'face_images')

        self.pub_cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        rospy.loginfo("Finished FaceRecogniser Init process...Ready")

    def _check_cam_ready(self):
        """
        Checks if the camera is ready by trying to get an image from the specified topic.
        """
        self.cam_image = None
        while self.cam_image is None and not rospy.is_shutdown():
            try:
                self.cam_image = rospy.wait_for_message(self.topic, Image, timeout=1.0)
                rospy.logdebug("Current " + self.topic + " READY=>" + str(self.cam_image))
            except:
                rospy.logerr("Current " + self.topic + " not ready yet, retrying.")

    def camera_callback(self, data):
        """
        Callback function for the camera topic. Stores the received image data.
        Args:
            data (sensor_msgs.msg.Image): Image data received from the camera.
        """
        self.cam_image = data

    def load_data(self):
        """
        Load data from the face_images directory. Processes the jpg files and extracts face representations.
        """
        self.known_face = []
        self.known_name = []
        files = fnmatch.filter(os.listdir(self.people_dir), '*.jpg')
        self._process_files(files)

    def _process_files(self, files):
        """
        Processes the given files by loading the images and extracting face representations.
        Args:
            files (list): A list of file names to process.
        """
        for file_name in files:
            img = dlib.load_rgb_image(os.path.join(self.people_dir, file_name))
            self._process_image(img, file_name)

    def _process_image(self, img, file_name):
        """
        Processes the given image by detecting faces and extracting the largest face.
        Args:
            img (np.array): The image to process.
            file_name (str): The name of the file (used for logging purposes).
        """
        img_detected = self.detector(img, 1)
        if not img_detected:
            rospy.loginfo("No face detected in image: " + file_name)
            return
        self._extract_biggest_face(img, img_detected, file_name)

    def _extract_biggest_face(self, img, img_detected, file_name):
        """
        Extracts the biggest face from the detected faces in an image.
        Args:
            img (np.array): The image containing the faces.
            img_detected (list): A list of detected faces.
            file_name (str): The name of the file (used for logging purposes).
        """
        bounding_boxes = [rect.width() * rect.height() for rect in img_detected]
        biggest_box_index = bounding_boxes.index(max(bounding_boxes))
        biggest_box = img_detected[biggest_box_index]
        self._align_and_add_face(img, biggest_box, file_name)

    def _align_and_add_face(self, img, box, file_name):
        """
        Aligns and adds the face represented by the given bounding box to the known faces.
        Args:
            img (np.array): The image containing the face.
            box (dlib.rectangle): The bounding box of the face.
            file_name (str): The name of the file (used for logging purposes).
        """
        img_shape = self.sp(img, box)
        align_img = dlib.get_face_chip(img, img_shape)
        img_rep = np.array(self.model.compute_face_descriptor(align_img))
        self.known_face.append(img_rep)
        self.known_name.append(file_name.split('.')[0].lower())

    def spin(self, velocidade=0.4):
        """
        Spins the robot at the given speed.
        Args:
            velocidade (float, optional): The speed at which to spin. Defaults to 0.4.
        """
        vel_cmd = Twist()
        vel_cmd.angular.z = velocidade
        self.pub_cmd_vel.publish(vel_cmd)

    def predict(self):
        """
        Predicts the bounding boxes in the current frame using the YOLO object detection model.
        Returns:
            results[0].boxes (list): A list of predicted bounding boxes.
        """
        small_frame = self.bridge_object.imgmsg_to_cv2(self.cam_image, desired_encoding="bgr8")
        small_frame = small_frame[:720, :]
        results = self.yolo.predict(source=small_frame, conf=0.4, device=0, classes=[56,57])
        print('Len boxes: ', len(results[0].boxes))
        return results[0].boxes

    def find_sit(self):
        """
        Finds an empty location by spinning and using object detection.
        Returns:
            center_place (float): The center of the found empty space.
        """
        boxes = self.predict()

        while len(boxes) == 0:
            print('First while')
            self.spin(0.4)
            boxes = self.predict()

        self.spin(0)

        while True:
            print('Second while')
            self.checked_places = []

            boxes = self.predict()
            if len(boxes) > 0:
                print('box > 0')
                self.spin(0)
                time.sleep(1)
                self.center_place = self.find_empty_place(boxes)

            if self.center_place is not None:
                print('break')
                break
            else:
                print('Spin and detect')
                self.spin(0.4)

        return self.center_place

    def find_empty_place(self, boxes):
        """
         Find an empty place not occupied by any recognized faces.

         Args:
             boxes (List[dlib.rectangles]): List of bounding boxes detected in the frame.

         Returns:
             center_place (float): The center of the free space.
         """
        print('Looking for an empty place')
        if len(boxes) == 0:
            print("No boxes detected, spinning...")
            self.spin(0.4)
            return None

        for box in boxes:
            center_place = self._get_center_place(box)

            if center_place is not None and center_place not in self.checked_places:
                self.checked_places.append(center_place)
                return center_place

        # If all places are checked, spin again to find a new place
        print("All places are checked, spinning...")
        self.spin(0.4)
        return None

    def _get_center_place(self, box):
        """
        Get the center of the free space for a given bounding box.

        Args:
            box (dlib.rectangles): Detected bounding box.

        Returns:
            center_place (float): The center of the free space.
        """
        obj_class = self.yolo.names[int(box.cls)]
        print(obj_class)

        if self.face_center:
            center_place = self._check_known_faces(box, obj_class)
        else:
            center_place = self._calculate_center_place(box, obj_class)

        return center_place

    def _check_known_faces(self, box, obj_class, center_place=None):
        """
        Check for known faces in the specified bounding box.

        Args:
            box (dlib.rectangles): Detected bounding box.
            obj_class (str): The class of the object ('chair' or 'couch').

        Returns:
            center_place (float): The center of the free space.
        """
        for i, face_name in enumerate(self.face_name):
            if face_name in self.known_name:
                center_place = self._calculate_center_place(box, obj_class, i, face_name)

        return center_place

    def _calculate_center_place(self, box, obj_class, index=None, face_name=None, center_place=None):
        """
        Calculate the center of the free space for a given object class.

        Args:
            box (dlib.rectangles): Detected bounding box.
            obj_class (str): The class of the object ('chair' or 'couch').

        Returns:
            center_place (float): The center of the free space.
        """
        box = box.xyxy[0]
        if index is not None and face_name is not None:
            print('box0:', box[0])
            print('box2:', box[2])
            print('center:', self.face_center[index])
        if obj_class == 'chair' and not any(box[0] < center_face < box[2] for center_face in self.face_center):
            center_place = (box[0] + box[2]) / 2
            print("lugar 0")
        elif obj_class == 'couch':
            media_x = (box[0] + box[2]) / 2
            if not any(box[0] < center_face < media_x for center_face in self.face_center):
                center_place = (box[0] + media_x) / 2
                print('lugar 1')
            elif not any(media_x < center_face < box[2] for center_face in self.face_center):
                center_place = (media_x + box[2]) / 2
                print('lugar 2')

        return center_place

    def recognize(self, small_frame):
        """
        Recognizes the faces in the given frame and draws bounding boxes around them.
        Args:
            small_frame (np.array): The frame in which to recognize faces.
        Returns:
            len(img_detected) (int): The number of detected faces.
        """
        self.face_center = []
        self.face_name = []

        img_detected = self.detector(small_frame, 1)

        if len(img_detected) == 0:
            rospy.loginfo("No face detected")
            self.find_sit()
            return len(img_detected)

        faces = dlib.full_object_detections()
        for detection in img_detected:
            faces.append(self.sp(small_frame, detection))
        align_img = dlib.get_face_chips(small_frame, faces)
        img_rep = np.array(self.model.compute_face_descriptor(align_img))

        for i in range(len(img_detected)):
            name = self._get_face_name(img_rep[i])
            self.face_name.insert(i, name)

        for i, rects in enumerate(img_detected):
            if self.face_name[i] in self.known_name:
                self._draw_rectangle(small_frame, rects, (0, 255, 0), self.face_name[i])
            else:
                self._draw_rectangle(small_frame, rects, (255, 0, 0), self.face_name[i])
            center_x = (rects.right() + rects.left()) / 2
            self.face_center.append(center_x)

        window = dlib.image_window()
        window.set_image(small_frame)
        cv2.imwrite(self.path_to_package + '/face_recogs/recog.jpg', small_frame)

        return len(img_detected)

    def _get_face_name(self, img_rep):
        """
        Gets the name of the face represented by the given representation by comparing it with known faces.
        Args:
            img_rep (np.array): The face representation to identify.
        Returns:
            name (str): The name of the recognized face, or 'Face' if not recognized.
        """
        name = 'Face'
        for j in range(len(self.known_face)):
            euclidean_dist = list(np.linalg.norm(self.known_face - img_rep, axis=1) <= 0.62)
            if True in euclidean_dist:
                fst = euclidean_dist.index(True)
                name = self.known_name[fst]
            else:
                continue
        return name

    def _draw_rectangle(self, frame, rects, color, name):
        """
                Draws a rectangle around the face in the frame and annotates it with the name.
                Args:
                    frame (np.array): The frame in which to draw the rectangle
        """
        cv2.rectangle(frame, (rects.left(), rects.top()), (rects.right(), rects.bottom()), color, 2)
        cv2.putText(frame, name, (rects.left(), rects.top()), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    def start(self, data, nome_main):
        nome_main = nome_main.lower()
        self.load_data()

        small_frame = self.bridge_object.imgmsg_to_cv2(data, desired_encoding="bgr8")

        num_faces = self.recognize(small_frame)

        name = 'Face'
        center = 0.0
        if nome_main == '':
            self.find_sit()
            print(self.center_place)
            self.recog = 1
        elif nome_main in self.face_name:
            name = nome_main
            index = self.face_name.index(nome_main)
            center = self.face_center[index]
            print(self.face_center)
            print(center)

            self.recog = 1

        return name, center, num_faces, self.center_place

    def handler(self, request):
        self.recog = 0
        while self.recog == 0:
            self.image_sub = rospy.Subscriber(self.topic, Image, self.camera_callback)
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
