#!/usr/bin/python
from importlib.util import module_for_loader
from colormap import rgb2hex
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from PIL import Image, ImageFile
from u2net_test import mask
from features_pkg.srv import features
from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset
from model import U2NETP # small version u2net 4.7 MB
from sensor_msgs.msg import Image as imgmsg
from cv_bridge import CvBridge
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import extcolors
import os
import argparse
import glob
import json
import rospy
import time


class bonusFeatures:

    def __init__(self):
        rospy.Service('features', features, self.handler)
        self.point = None
        self.cint = None
        self.knee = None
        self.neck = None 
        self.BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}
        self.POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]
        self.topic = "/usb_cam/image_raw"
        self.img_sub = rospy.Subscriber(self.topic,imgmsg,self.camera_callback)

    def body_points(self,frame):
        points = []
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        for i in range(len(self.BODY_PARTS)): 
            heatMap = out[0, i, :, :]

            _, conf, _, point = cv2.minMaxLoc(heatMap)
            x = (frameWidth * point[0]) / out.shape[3]
            y = (frameHeight * point[1]) / out.shape[2]
            points.append((int(x), int(y)) if conf > 0.2 else None)

        for pair in self.POSE_PAIRS:
            partFrom = pair[0]
            partTo = pair[1]
            assert (partFrom in self.BODY_PARTS)
            assert (partTo in self.BODY_PARTS)

            idFrom = self.BODY_PARTS[partFrom]
            idTo = self.BODY_PARTS[partTo]

            if points[idFrom] and points[idTo]:
                cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
                cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

            t, _ = net.getPerfProfile()
            freq = cv2.getTickFrequency() / 1000
            lx = cv2.putText(frame, '%.2fms' % (t / freq), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            cv2.imwrite('results/pose_points.png', lx)
            with open('points.json', 'w') as f:
                json.dump(points, f)
            break

    def creating_mask(self,frame):
        rospy.loginfo("Starting mask")
        output = mask()
        output = load_img(output)
        rescale_val = 255
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

        rem_back_scaled.save('results/removed_background.png')
        rospy.loginfo("Sucessfully created mask!")
        out_layer = out_img[:,:,1]
        y_starts = [np.where(out_layer.T[i]==1)[0][0] if len(np.where(out_layer.T[i]==1)[0])!=0 else out_layer.T.shape[0]+1 for i in range(out_layer.T.shape[0])]
        
        global starty
        starty = min(y_starts)

    def height_estimate(self,distance, height):
        height = 1280 - height
        distance += 32
        camera_image_height = 1.48 * distance
        if height < 640:
            subject_height = 640 - height
            hf = 1.52 - (((subject_height*camera_image_height)/1280)/100)
        else:
            subject_height = height - 640
            hf = (((subject_height*camera_image_height)/1280)/100) + 1.52
        return hf

    def ifmask(path):
	        
        model = load_model('/home/bibo/catkin_dev/src/hera_face/features_pkg/src/mask_detector.model')
        openimage = cv2.imread(path)
        image = cv2.resize(openimage,(224,224))
        image = np.reshape(image,[1,224,224,3]) 
        predict = model.predict(image)[0]

        if predict[0] > predict[1]:
            return "Mask"
        else:
            return "No Mask"
	
    def pose_points(self,frame):
        rospy.loginfo("finding pose points")
        null = 'null'
        
        inWidth = 480
        inHeight = 640

        net = cv2.dnn.readNetFromTensorflow("/home/bibo/catkin_dev/src/hera_face/features_pkg/src/graph_opt.pb")

        net.setInput(
            cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False)) 
        out = net.forward()
        out = out[:, :19, :, :]  

        assert (len(self.BODY_PARTS) == out.shape[1]) 
        
        
        self.creating_mask(frame)
        

        self.body_points(frame)

        with open('points.json', 'r') as f:
            self.point = json.load(f)

        try:
            if self.point[1][1] != null:
                self.neck = self.point[1][1] 
        except: 
            self.neck = starty-25

        try:
            if self.point[12][1] != null:
                self.knee = self.point[12][1]
        except:
            try:
                if self.point[9][1] != null:
                    self.knee = point[9][1]
            except:
                self.knee = null
        
        try:
            if self.point[8][1] != null:
                self.cint = self.point[8][1] +25
        except:
            try:
                if self.point[11][1] != null:
                    self.cint = self.point[11][1]+25
            except:
                self.cint = ((frame.shape[1]-starty)/12)*7

        way = glob.glob('images/*')
        for py_file in way:
            try:
                os.remove(py_file)
            except OSError as e:
                print(f"Error:{ e.strerror}")
                
        modelo = cv2.imread('results/removed_background.png')

        for x in range(3):
            if x == 0:
                foto = modelo[self.neck - 25:self.cint, 0:]
                cv2.imwrite('images/torso.png', foto)  
            elif x == 1:
                if self.knee != null:
                    print("KNEE", self.knee)
                    print("CINT:", self.cint)
                    foto = modelo[self.cint - 25:self.knee, 0:]
                    cv2.imwrite('images/pernas.png', foto)
                else: 
                    foto = modelo[self.cint - 25:, 0:]
                    cv2.imwrite('images/pernas.png', foto)
            else:
                foto = modelo[:self.neck - 10]
                cv2.imwrite('images/cabeca.png', foto)    

    def color(self,path):
        ImageFile.LOAD_TRUNCATED_IMAGES = True 
        plt.figure(figsize=(9, 9))
        img = plt.imread(path)
        plt.imshow(img)
        plt.axis('off')

        colors_x = extcolors.extract_from_path(img, tolerance=12, limit=12)

        rgb = (colors_x[0][0][0])

        if rgb == (0, 0, 0):
            rgb = (colors_x[0][1][0])
        if rgb == (255, 255, 255):
            return 'White'
        if rgb < (45, 45, 45):
            return 'Black'
        elif rgb[0] == rgb[1] and rgb[1] == rgb[2] and rgb[0] == rgb[2]:
            return 'Grey'

        elif rgb[0] > rgb[1] and rgb[0] > rgb[2]:
            if rgb[0] > 209 and rgb[1] > 179 and rgb[2] > 134 and rgb != (255, 192, 203):
                return 'Beige'

            elif (rgb == (184, 134, 11) or rgb == (189, 83, 107) or rgb == (139, 69, 19) or rgb == (160, 82, 45) or rgb == (
            188, 143, 143)) or rgb[0] > 204 and rgb[1] > 104 and rgb[2] < 144:
                return 'Brown'

            elif rgb[0] > 204 and rgb[1] < 193 and rgb[2] > 91:
                return 'Pink'

            elif rgb == (255, 140, 0) or rgb == (255, 165, 0):
                return 'Orange'

            elif rgb == (255, 215, 0):
                return 'Gold'
            elif rgb == (189, 83, 107):
                return 'Green'
            else:
                return 'Red'

        elif rgb[1] > rgb[0] and rgb[1] > rgb[2] or rgb == (47, 79, 79):
            if rgb == (133, 130, 111) or rgb == (124, 125, 111):
                return 'Beige'
            return 'Green'

        elif rgb[2] > rgb[1] and rgb[2] > rgb[0] or rgb == (0, 255, 255) or rgb == (0, 139, 139) or rgb == (0, 128, 128):
            if rgb[0] > 122 and rgb[1] < 113 and rgb[2] > 203 or rgb == (128, 0, 128) or rgb == (75, 0, 130):
                return 'Purple'
            else:
                return 'Blue'

        elif rgb == (128, 128, 0):
            return 'Green'
        elif rgb == (255, 255, 0):
            return 'Yellow'
        elif rgb == (255, 0, 255) or rgb == (238, 130, 238) or rgb == (218, 112, 214) or rgb == (221, 160, 221):
            return 'Pink'


    def features(self,distance):
        self.bridge = CvBridge()
        frame = self.bridge.imgmsg_to_cv2(self.cam_image,desired_encoding='bgr8')
        time.sleep(1)
        cv2.imwrite('base/img.png', frame)    

        self.pose_points(frame)

        for parts in glob.glob('images/*'):

            if parts == 'images/cabeca.png':
                mask = ifmask('images/cabeca.png')

            output_color = self.color(parts)
            body_colors.append(output_color)
        # Return the list of colors
        height_estimate(distance, starty)
        return mask,str(body_colors[0]),str(body_colors[1]), height_estimate

    def camera_callback(self,data):
        self.cam_image = data

    def handler(self, request):
        self.recog = 0
        rospy.loginfo("Service called!")
        rospy.loginfo("Requested..")
        while self.recog == 0:
            self.img_sub = rospy.Subscriber(self.topic,imgmsg,self.camera_callback)
            mask, shirt, pants, height = self.features(request)
            self.rate.sleep()
            return mask, shirt, pants, height
        cv2.destroyAllWindows()
    

if __name__ == "__main__":
    rospy.init_node('feature_bonus', log_level=rospy.INFO)
    rospy.loginfo("Service started!")

    bonusFeatures()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

    





