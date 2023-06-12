#!/usr/bin/env python3
from pyexpat import model
import sys
import os
from turtle import back
from unicodedata import name
import rospy
from sensor_msgs.msg import Image
import rospkg
import fnmatch
import cv2
import time
from cv_bridge import CvBridge, CvBridgeError
from hera_face.srv import face_list
from ultralytics import YOLO
import dlib
import numpy as np

class FaceRecog():
    # cuidado para nao ter imagem com tamanhos diferentes ou cameras diferentes, pois o reconhecimento nao vai funcionar
    recog = 0
    def __init__(self):
        rospy.Service('face_recog', face_list, self.handler)
        rospy.loginfo("Start FaceRecogniser Init process...")
        # get an instance of RosPack with the default search paths
        self.rate = rospy.Rate(5)
        rospack = rospkg.RosPack()
        # get the file path for my_face_recogniser
        self.path_to_package = rospack.get_path('hera_face')
        self.yolo = YOLO(self.path_to_package+'/src/coco.pt')
        self.bridge_object = CvBridge()
        rospy.loginfo("Start camera suscriber...")
        self.topic = "/zed_node/left_raw/image_raw_color"
        self._check_cam_ready()
        self.image_sub = rospy.Subscriber(self.topic,Image,self.camera_callback)
        rospy.loginfo("Finished FaceRecogniser Init process...Ready")

    def load_data(self):
        self.detector = dlib.get_frontal_face_detector()
        self.sp = dlib.shape_predictor(self.path_to_package+ "/src/shape_predictor_5_face_landmarks.dat")
        self.model  = dlib.face_recognition_model_v1(self.path_to_package+"/src/dlib_face_recognition_resnet_model_v1.dat")
        self.people_dir = self.path_to_package+'/face_images/'
        files = fnmatch.filter(os.listdir(self.people_dir), '*.jpg')

        self.known_face = []
        self.known_name = []
        for f in range(0, len(files)):
            img = dlib.load_rgb_image(self.people_dir + files[f])
            img_detected = self.detector(img, 1)
            bouding_boxes = []
            for i, rects in enumerate(img_detected):
                area = rects.top() * rects.left()
                bouding_boxes.append(area)
            biggest_box = bouding_boxes.index(max(bouding_boxes))
            img_shape = self.sp(img, img_detected[biggest_box])
            align_img = dlib.get_face_chip(img, img_shape)
            img_rep = np.array(self.model.compute_face_descriptor(align_img))
            if len(img_detected) > 0:
                self.known_face.append(img_rep)
                self.known_name.append(files[f].split('.')[0])
                break 
            else:
                rospy.loginfo("No face detected in image: " + files[f])
                break

    def find_sit(self, small_frame):
        place = False
        results = self.yolo.predict(source=small_frame, conf=0.7, device=0, classes=[56,57])
        boxes = results[0].boxes
        while (not boxes.cls.index(56) or not boxes.cls.index(57)) and not place:
            #SPIN
            place = self.find_empty_place(boxes)
            pass
        self.place = place          

    def find_empty_place(self, boxes):
        place = False
        for i, c in enumerate(boxes.cls):
            box = boxes[i].xyxy[0]
            obj_class = self.yolo.names[int(c)]
            for i in range(0, len(self.face_center)):
                if self.face_name[i] in self.known_name:
                    if obj_class == 'chair':
                        if (box[0] < self.face_center[i] < box[2]):
                            place = True     
                    elif obj_class == 'couch':
                        media_x = (box[0] + box[2]) / 2
                        if (box[0] < self.face_center[i] < media_x):
                            place = True
                        elif (media_x < self.face_center[i] < box[2]):
                            place = True
        return place

    def _check_cam_ready(self):
      self.cam_image = None
      while self.cam_image is None and not rospy.is_shutdown():
         try:
               self.cam_image = rospy.wait_for_message(self.topic, Image, timeout=1.0)
               rospy.logdebug("Current "+self.topic+" READY=>" + str(self.cam_image))
         except:
               rospy.logerr("Current "+self.topic+" not ready yet, retrying.")

    def camera_callback(self,data):
        self.cam_image = data

    def recognize(self, data, nome_main):
        self.load_data()
        #Get image from topic
        small_frame = self.bridge_object.imgmsg_to_cv2(data, desired_encoding="bgr8")
        time.sleep(1)   
    
        face_center = []
        self.face_name = []
        img_detected = self.detector(small_frame, 1)
        #Check if there are people
        if len(img_detected) == 0:
            rospy.loginfo("No face detected")
            return '', 0.0, len(img_detected), 404
        else:
            print("Face Detectada", img_detected)
            faces = dlib.full_object_detections()
            for detection in img_detected:
                faces.append(self.sp(small_frame, detection))
            align_img = dlib.get_face_chips(small_frame, faces)                    
            img_rep = np.array(self.model.compute_face_descriptor(align_img))
        #--------------------------------------------------------------------------------
        #Match known faces with current faces
            for i in range(0, len(img_detected)):
                name = 'Face'
                for _ in range(0, len(self.known_face)):      
                    euclidean_dist = list(np.linalg.norm(self.known_face - img_rep[i], axis=1) <= 0.6)
                    if True in euclidean_dist:
                        fst = euclidean_dist.index(True)
                        name = self.known_name[fst]
                    else:
                        continue
                self.face_name.insert(i, name)
        #--------------------------------------------------------------------------
        #Plot boxes
            for i, rects in enumerate(img_detected):
                if self.face_name[i] in self.known_name:
                    cv2.rectangle(small_frame, (rects.left(), rects.top()), (rects.right(), rects.bottom()), (0, 255, 0), 2)
                    cv2.putText(small_frame, self.face_name[i], (rects.left(), rects.top()), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    cv2.rectangle(small_frame, (rects.left(), rects.top()), (rects.right(), rects.bottom()), (255, 0, 0), 2)
                    cv2.putText(small_frame, self.face_name[i], (rects.left(), rects.top()), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                center_x = (rects.right() + rects.left())/2
                self.face_center.append(center_x)
            
            window = dlib.image_window()
            window.set_image(small_frame)
            cv2.imwrite(self.path_to_package+'/face_recogs/recog.jpg', small_frame)

            print("Face Recognized: ", self.face_name)
            print("Face centers: ", self.face_center)
            print("People in the photo: ", len(img_detected))
        #--------------------------------------------------------------------------
            if nome_main == '':
                for i in self.known_name:
                    if i in self.face_name:
                        self.find_sit(small_frame)
                        center = face_center[self.face_name.index(i)]
                        name = self.face_name[self.face_name.index(i)]
                        print("Pessoa conhecida encontrada")
                    else:
                        name = 'face'
                        center = '0.0'
                        self.recog = 1
            elif nome_main in self.face_name:
                center = face_center[self.face_name.index(nome_main)]
                name = self.face_name[self.face_name.index(nome_main)]
                print("Pessoa encontrada")
                self.recog = 1
            else:
                self.recog = 0
                name = 'face'
                center = '0.0'
            return name, center, len(img_detected), self.place

    def handler(self, request):
        self.recog = 0
        while self.recog == 0:
            self.image_sub = rospy.Subscriber(self.topic,Image,self.camera_callback)
            name, center, num, empty = self.recognize(self.cam_image, request.name)
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
