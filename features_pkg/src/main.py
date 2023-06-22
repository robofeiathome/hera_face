#!/usr/bin/python
from importlib.util import module_for_loader
from colormap import rgb2hex
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from PIL import Image, ImageFile
from u2net_test import mask
from features_pkg.srv import Features
from sensor_msgs.msg import Image as imgmsg
from cv_bridge import CvBridge
from colorthief import ColorThief
import webcolors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import argparse
import glob
import json
import extcolors
import rospy
import time
import dlib
import statistics

BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
            "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
            "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
            "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}
POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
            ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
            ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
            ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
            ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]
class bonusFeatures:

    def __init__(self):
        self.point = None
        self.cint = None
        self.knee = None
        self.neck = None 
        self.bridge = CvBridge()

        self.topic = "/usb_cam/image_raw"
        self.rate = rospy.Rate(5)
        rospy.Service('features', Features, self.handler)
        self.img_sub = rospy.Subscriber(self.topic,imgmsg,self.camera_callback)

    def body_points(self,frame):
        inWidth = 480
        inHeight = 640

        net = cv2.dnn.readNetFromTensorflow("/home/bibo/catkin_dev/src/hera_face/features_pkg/src/graph_opt.pb")

        net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
        out = net.forward()
        out = out[:, :19, :, :] 
        assert (len(BODY_PARTS) == out.shape[1]) 

        points = []
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        for i in range(len(BODY_PARTS)): 
            heatMap = out[0, i, :, :]

            _, conf, _, point = cv2.minMaxLoc(heatMap)
            x = (frameWidth * point[0]) / out.shape[3]
            y = (frameHeight * point[1]) / out.shape[2]
            points.append((int(x), int(y)) if conf > 0.2 else None)

        for pair in POSE_PAIRS:
            partFrom = pair[0]
            partTo = pair[1]
            assert (partFrom in BODY_PARTS)
            assert (partTo in BODY_PARTS)

            idFrom = BODY_PARTS[partFrom]
            idTo = BODY_PARTS[partTo]

            if points[idFrom] and points[idTo]:
                cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
                cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

            t, _ = net.getPerfProfile()
            freq = cv2.getTickFrequency() / 1000
            lx = cv2.putText(frame, '%.2fms' % (t / freq), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            cv2.imwrite('src/hera_face/features_pkg/src/results/pose_points.png', lx)
            with open('src/hera_face/features_pkg/src/points.json', 'w') as f:
                json.dump(points, f)
            break

    def creating_mask(self,frame):
        output = mask()
        output = load_img(output)
        rescale_val = 1
        out_img = img_to_array(output) / rescale_val
        THRESHOLD = 0.2
        out_img[out_img > THRESHOLD] = 1
        out_img[out_img <= THRESHOLD] = 0
        shape = out_img.shape
        a_layer_init = np.ones(shape=(shape[0], shape[1], 1))
        mul_layer = np.expand_dims(out_img[:, :, 0], axis=2)
        a_layer = mul_layer * a_layer_init
        rgba_out = np.append(out_img, a_layer, axis=2)
        
        inp_img = img_to_array(frame)
        inp_img = cv2.resize(inp_img, (640, 480))

        a_layer = np.ones(shape=(shape[0], shape[1], 1))
        rgba_inp = np.append(inp_img, a_layer, axis=2)
        rem_back = (rgba_inp * rgba_out)
        rem_back_scaled = Image.fromarray((rem_back * rescale_val).astype('uint8'), 'RGBA')

        rem_back_scaled.save('src/hera_face/features_pkg/src/results/removed_background.png')
        rospy.loginfo("Sucessfully created mask!")
        out_layer = out_img[:,:,1]
        y_starts = [np.where(out_layer.T[i]==1)[0][0] if len(np.where(out_layer.T[i]==1)[0])!=0 else out_layer.T.shape[0]+1 for i in range(out_layer.T.shape[0])]
        
        self.starty = min(y_starts)

    def ifmask(self,path):
	        
        model = load_model('/home/bibo/catkin_dev/src/hera_face/features_pkg/src/mask_detector.model')
        openimage = cv2.imread(path)
        image = cv2.resize(openimage,(224,224))
        image = np.reshape(image,[1,224,224,3]) 
        predict = model.predict(image)[0]

        if predict[0] < predict[1]:
            return "Mask"
        else:
            return "No Mask"
    
    def ifglasses(self,path):

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('src/hera_face/features_pkg/src/shape_predictor_68_face_landmarks.dat')
        img = dlib.load_rgb_image(path)
        face = detector(img,1)
        print(face)
        while len(face)<=0:
            rospy.logwarn("No faces found")
            rect = detector(img,1)
            
        rect = detector(img)[0]
        sp = predictor(img, rect)
        landmarks = np.array([[p.x, p.y] for p in sp.parts()])
        nose_bridge_x = []
        nose_bridge_y = []
        for i in [28,29,30,31,33,34,35]:
            nose_bridge_x.append(landmarks[i][0])
            nose_bridge_y.append(landmarks[i][1])
                
        ### x_min and x_max
        x_min = min(nose_bridge_x)
        x_max = max(nose_bridge_x)### ymin (from top eyebrow coordinate),  ymax
        y_min = landmarks[20][1]
        y_max = landmarks[31][1]
        img2 = Image.open(path)
        img2 = img2.crop((x_min,y_min,x_max,y_max))

        img_blur = cv2.GaussianBlur(np.array(img2),(3,3), sigmaX=0, sigmaY=0)
        edges = cv2.Canny(image =img_blur, threshold1=100, threshold2=200)

        #center strip
        edges_center = edges.T[(int(len(edges.T)/2))]

        if 255 in edges_center:
            return 'Glasses'
        else:
            return 'No glasses'
	
    def pose_points(self,frame):
        null = 'null'
        
        self.creating_mask(frame)

        self.body_points(frame)

        with open('src/hera_face/features_pkg/src/points.json', 'r') as f:
            self.point = json.load(f)

        try:
            if self.point[1][1] != null:
                self.neck = self.point[1][1] 
        except: 
            self.neck = self.starty-25

        try:
            if self.point[12][1] != null:
                self.knee = self.point[12][1]
        except:
            try:
                if self.point[9][1] != null:
                    self.knee = self.point[9][1]
            except:
                self.knee = null
        
        try:
            if self.point[8][1] != null:
                self.cint = self.point[8][1] 
        except:
            try:
                if self.point[11][1] != null:
                    self.cint = self.point[11][1]
            except:
                self.cint = int(((frame.shape[1]-self.starty)/12)*7)

        way = glob.glob('src/hera_face/features_pkg/src/images/*')
        for py_file in way:
            try:
                os.remove(py_file)
            except OSError as e:
                rospy.logerr(f"Error:{ e.strerror}")
  
                
        modelo = cv2.imread('src/hera_face/features_pkg/src/results/removed_background.png')
        modelo = modelo[0:480,200:440]

        for x in range(3):
            if x == 0:
                foto = modelo[self.neck - 25:self.cint, 0:]
                cv2.imwrite('src/hera_face/features_pkg/src/images/torso.png', foto)  
            elif x == 1:
                if self.knee != null:
                    foto = modelo[self.cint - 25:self.knee, 0:]
                    cv2.imwrite('src/hera_face/features_pkg/src/images/pernas.png', foto)
                else: 
                    foto = modelo[self.cint - 25:, 0:]
                    cv2.imwrite('src/hera_face/features_pkg/src/images/pernas.png', foto)
            else:
                foto = modelo[:self.neck - 10]
                cv2.imwrite('src/hera_face/features_pkg/src/images/cabeca.png', foto)    

    def color(self,path):
        for i in range(2):
            color = self.findColor(path,i)
            print(color)
            if 'black' in color:
                if(self.isitblack(path, i)):  
                    return 'Black'
            else:
                return color

    def findColor(self,path,index):
        color_thief = ColorThief(path)
        dominant_color = color_thief.get_palette(color_count=2)
        closest_color = None
        min_distance = float('inf')
        for color_name, rgb in webcolors.CSS3_NAMES_TO_HEX.items():
            css3_rgb = webcolors.hex_to_rgb(rgb)

            distance = self.delta_e_cie76(dominant_color[index], css3_rgb)

            if distance < min_distance:
                min_distance = distance
                closest_color = color_name
        return closest_color

    def delta_e_cie76(self,color1, color2):
        r_mean = (color1[0] + color2[0]) / 2
        delta_r = color1[0] - color2[0]
        delta_g = color1[1] - color2[1]
        delta_b = color1[2] - color2[2]
        return ((2 + (r_mean / 256)) * delta_r ** 2 + 4 * delta_g ** 2 + (2 + ((255 - r_mean) / 256)) * delta_b ** 2) ** 0.5

    def isitblack(self,path,index):
        colors_x, pixel_count = extcolors.extract_from_path(path)
        pclist = []
        for color in colors_x:
            rgb = color[0]
            count = color[1]
            percentage = (count / pixel_count) * 100
            pclist.append(percentage)
        #print(pclist[index])
        if (pclist[index])>85:
            return True
        else:
            return False

    
    def features(self):
        frame = self.bridge.imgmsg_to_cv2(self.cam_image,desired_encoding='rgb8')
        time.sleep(1)
        
        cv2.imwrite('src/hera_face/features_pkg/src/base/img.png', frame)    
        
        self.pose_points(frame)
        body_colors = []
        for parts in glob.glob('src/hera_face/features_pkg/src/images/*'):

            if parts == 'src/hera_face/features_pkg/src/images/cabeca.png':
                mask = self.ifmask('src/hera_face/features_pkg/src/images/cabeca.png')
                glasses = self.ifglasses('src/hera_face/features_pkg/src/images/cabeca.png')
                    
            output_color = self.color(parts)
            body_colors.append(output_color)

        return mask,glasses,str(body_colors[0]),str(body_colors[1])

    def camera_callback(self,data):
        self.cam_image = data

    def handler(self, request):
        self.recog = 0
        rospy.loginfo("Service called!")
        rospy.loginfo("Requested..")
        time.sleep(2)
        while self.recog == 0:
            self.img_sub = rospy.Subscriber(self.topic,imgmsg,self.camera_callback)
            mask, glasses, shirt, pants = self.features()
            self.rate.sleep()
            return mask, glasses, shirt, pants
        cv2.destroyAllWindows()
    
if __name__ == "__main__":
    rospy.init_node('feature_bonus', log_level=rospy.INFO)
    rospy.loginfo("Service started!")

    bonusFeatures()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

    





