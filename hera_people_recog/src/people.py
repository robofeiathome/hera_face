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
        self.bridge_object = CvBridge()
        rospy.loginfo("Start camera suscriber...")
        self.topic = "/zed2i/zed_node/left_raw/image_raw_color"
        #self._check_cam_ready()
        #self.image_sub = rospy.Subscriber(self.topic,Image,self.camera_callback)
        rospy.loginfo("Finished FaceRecogniser Init process...Ready")

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

    def recognize(self, data, nome_main, place):
        place = False if place == '' else True

        detector = dlib.get_frontal_face_detector()
        sp = dlib.shape_predictor(self.path_to_package+ "/src/shape_predictor_5_face_landmarks.dat")
        model  = dlib.face_recognition_model_v1(self.path_to_package+"/src/dlib_face_recognition_resnet_model_v1.dat")
        self.people_dir = self.path_to_package+'/face_images/'
        print(self.people_dir)
        files = fnmatch.filter(os.listdir(self.people_dir), '*.jpg')

        known_face = []
        known_name = []
        for f in range(0, len(files)):
            for j in range(0,100):
                img = dlib.load_rgb_image(self.people_dir + files[f])
                img_detected = detector(img, 1)
                img_shape = sp(img, img_detected[0])
                align_img = dlib.get_face_chip(img, img_shape)
                img_rep = np.array(model.compute_face_descriptor(align_img))
                if len(img_detected) > 0:
                    known_face.append(img_rep)
                    known_name.append(files[f].split('.')[0])
                    break 
                else:
                    rospy.loginfo("No face detected in image: " + files[f])
                    break

        #small_frame = self.bridge_object.imgmsg_to_cv2(data, desired_encoding="bgr8")
        small_frame = cv2.imread(self.path_to_package+'/src/notcrop.jpg')
        time.sleep(1)   
        # Get a reference to webcam #0 (the default one)
        # We select bgr8 because its the OpneCV encoding by default
    
        # robot vision
        face_center = []
        face_name = []
        img_detected = detector(small_frame, 1)
        # se nao detectar ninguem
        if len(img_detected) == 0:
            rospy.loginfo("No face detected")
            return '', 0.0, len(img_detected)
        else:
            print("Face Detectada", img_detected)
            faces = dlib.full_object_detections()
            for detection in img_detected:
                faces.append(sp(small_frame, detection))
            
            align_img = dlib.get_face_chips(small_frame, faces)                    
            img_rep = np.array(model.compute_face_descriptor(align_img))
        #--------------------------------------------------------------------------------
            for i in range(0, len(img_detected)):
                name = 'Face'
                for k in range(0, len(known_face)):      
                    euclidean_dist = list(np.linalg.norm(known_face - img_rep[i], axis=1) <= 0.6)
                    if True in euclidean_dist:
                        fst = euclidean_dist.index(True)
                        name = known_name[fst]
                    else:
                        continue
                 
                face_name.insert(i, name)
        #--------------------------------------------------------------------------
            for i, rects in enumerate(img_detected):
                if face_name[i] in known_name:
                    cv2.rectangle(small_frame, (rects.left(), rects.top()), (rects.right(), rects.bottom()), (0, 255, 0), 2)
                    cv2.putText(small_frame, face_name[i], (rects.left(), rects.top()), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    cv2.rectangle(small_frame, (rects.left(), rects.top()), (rects.right(), rects.bottom()), (255, 0, 0), 2)
                    cv2.putText(small_frame, face_name[i], (rects.left(), rects.top()), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                center_x = (rects.right() + rects.left())/2
                face_center.append(center_x)
            
            window = dlib.image_window()
            window.set_image(small_frame)
            cv2.imwrite(self.path_to_package+'/face_recogs/recog.jpg', small_frame)

            print("Face Recognized: ", face_name)
            print("Face centers: ", face_center)
            print("Pessoas na foto: ", len(img_detected))
            if place:
                h, w, _= small_frame.shape
                first_line = 734
                second_line = 1039
                third_line = 1274
                fourth_line = 1544
                last_line = 1844

                cv2.line(small_frame, (second_line, 0), (second_line, h), (0,0,255), thickness=2)
                cv2.line(small_frame, (first_line, 0), (first_line, h), (0,0,255), thickness=2)
                cv2.line(small_frame, (third_line, 0), (third_line, h), (0,0,255), thickness=2)
                cv2.line(small_frame, (fourth_line, 0), (fourth_line, h), (0,0,255), thickness=2)
                cv2.line(small_frame, (last_line, 0), (last_line, h), (0,0,255), thickness=2)
            
                places = [False, False, False, False]
                for i, rects in enumerate(img_detected):
                    if first_line < face_center[i] < second_line:
                        places[0] = True
                    elif second_line < face_center[i] < third_line:
                        places[1] = True
                    elif third_line < face_center[i] < fourth_line:
                        places[2] = True
                    elif fourth_line < face_center[i] < last_line:
                        places[3] = True
                    else:
                        print('All places are empty')

                empty_place = places.index(False)
                print(places)
                print(empty_place)
                    
                    
#                cv2.imwrite(self.path_to_package+'/face_recogs/crop.jpg', small_frame)
            #---------------------------------------------------------------------
            if nome_main == '':
                name = 'face'
                center = '0.0'
                self.recog = 1
            elif nome_main in face_name:
                center = face_center[face_name.index(nome_main)]
                name = face_name[face_name.index(nome_main)]
                print("Pessoa encontrada")
                self.recog = 1
            else:
                self.recog = 0
                name = 'face'
                center = '0.0'
            return name, center, len(img_detected)

    def handler(self, request):
        self.recog = 0
        if request.name == '':
            while self.recog == 0:
                self.image_sub = rospy.Subscriber(self.topic,Image,self.camera_callback)
                
                name, center, num = self.recognize(0, request.name, request.place)
                self.rate.sleep()
            
                return name, float(center), num
        else:
            # retornar somente o nome da pessoa e a posicao em center_x
            while self.recog == 0:
                self.image_sub = rospy.Subscriber(self.topic,Image,self.camera_callback)
                
                name, center, num = self.recognize(0, request.name, request.place)#self.cam_image
                self.rate.sleep()
        
                return name, float(center), num

        cv2.destroyAllWindows()
    
if __name__ == '__main__':
    rospy.init_node('face_recog', log_level=rospy.INFO)
    FaceRecog()

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass