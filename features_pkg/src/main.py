from importlib.util import module_for_loader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
import cv2
import extcolors
import os
from colormap import rgb2hex
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from u2net_test import mask
import glob
import json
from mask_detect import ifmask
from height_estimate import height_estimate
import time 

class principal:

    def __init__(self,fr):
        self.point = None
        self.cint = None
        self.knee = None
        self.neck = None 
        self.frame = fr
        self.features(0.5)


    def creating_mask(self):
        pf = glob.glob('/home/bibo/catkin_ws/src/hera_face/features_pkg/src/base/*')
        img_path = pf[0]
        output = mask(self.frame)
        output = load_img(output)
        RESCALE = 255
        out_img = img_to_array(output) / RESCALE
        THRESHOLD = 0.2
        out_img[out_img > THRESHOLD] = 1
        out_img[out_img <= THRESHOLD] = 0
        shape = out_img.shape
        #print("Shape: ", shape)
        a_layer_init = np.ones(shape=(shape[0], shape[1], 1))
        mul_layer = np.expand_dims(out_img[:, :, 0], axis=2)
        a_layer = mul_layer * a_layer_init
        rgba_out = np.append(out_img, a_layer, axis=2)

        
        original_image_path = img_path
        original_image = load_img(original_image_path)
        inp_img = img_to_array(original_image)
        inp_img = cv2.resize(inp_img, (640, 480))

        # since the output image is rgba, convert this also to rgba, but with no transparency
        a_layer = np.ones(shape=(shape[0], shape[1], 1))
        #print("Shape 1", a_layer.shape)
        #print("Shape 2", inp_img.shape)
        rgba_inp = np.append(inp_img, a_layer, axis=2)
        #print("Shape 3", rgba_inp.shape)
        #print("Shape 4", rgba_out.shape)
        # simply multiply the 2 rgba images to remove the backgound
        rem_back = (rgba_inp * rgba_out)
        rem_back_scaled = Image.fromarray((rem_back * RESCALE).astype('uint8'), 'RGBA')
        # save the resulting image to colab

        rem_back_scaled.save('results/removed_background.png')

        out_layer = out_img[:,:,1]
        y_starts = [np.where(out_layer.T[i]==1)[0][0] if len(np.where(out_layer.T[i]==1)[0])!=0 else out_layer.T.shape[0]+1 for i in range(out_layer.T.shape[0])]
        
        global starty
        starty = min(y_starts)

    def pose_points(self,frame):
        null = 'null'
        
        
        BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}
        #Dictionary of Body parts and Pose pairs returned from detection#
        POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]
        inWidth = 480
        inHeight = 640

        #Load the dnn#
        net = cv2.dnn.readNetFromTensorflow("/home/bibo/catkin_ws/src/hera_face/features_pkg/src/graph_opt.pb")

        frameWidth = self.frame.shape[1]
        frameHeight = self.frame.shape[0]

        #Checks predictions and filters for 19 elements#
        net.setInput(
            cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False)) #Transforms into binary 
        out = net.forward()
        out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

        assert (len(BODY_PARTS) == out.shape[1]) #Confirms that num of body parts = num returned by the dnn, if it doesent, the code is killed 

        points = []
        for i in range(len(BODY_PARTS)): #Checking each bodypart
            heatMap = out[0, i, :, :]

            # Originally, we try to find all the local maximums. To simplify a sample
            # we just find a global one. However only a single pose at the same time
            # could be detected this way.
            _, conf, _, point = cv2.minMaxLoc(heatMap)
            x = (frameWidth * point[0]) / out.shape[3]
            y = (frameHeight * point[1]) / out.shape[2]
            # Add a point if it's confidence is higher than threshold.
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
                cv2.ellipse(self.frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
                cv2.ellipse(self.frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

            t, _ = net.getPerfProfile()
            freq = cv2.getTickFrequency() / 1000
            lx = cv2.putText(self.frame, '%.2fms' % (t / freq), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            cv2.imwrite('results/pose_points.png', lx)
            with open('points.json', 'w') as f:
                json.dump(points, f)

            break
        self.creating_mask()

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
                self.cint = (self.frame.shape[1]/12)*7

        
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
        print("Saved points!!")



    def color(self):
        ImageFile.LOAD_TRUNCATED_IMAGES = True  # Permitir que imagens corrompidas sejam usadas
        segm_image = xxxxx  # Caminho da imagem
        # Mostrar imagem
        plt.figure(figsize=(9, 9))
        img = plt.imread(self.frame)
        plt.imshow(img)
        plt.axis('off')

        colors_x = extcolors.extract_from_path(self.frame, tolerance=12, limit=12)

        rgb = (colors_x[0][0][0])

        if rgb == (0, 0, 0):
            rgb = (colors_x[0][1][0])
            #print(rgb)
        if rgb == (255, 255, 255):
            #print('White')
            return 'White'
        if rgb < (45, 45, 45):
            #print('Black')
            return 'Black'
        elif rgb[0] == rgb[1] and rgb[1] == rgb[2] and rgb[0] == rgb[2]:
            #print('Grey')
            return 'Grey'

        elif rgb[0] > rgb[1] and rgb[0] > rgb[2]:
            if rgb[0] > 209 and rgb[1] > 179 and rgb[2] > 134 and rgb != (255, 192, 203):
                #print('Beige')
                return 'Beige'

            elif (rgb == (184, 134, 11) or rgb == (189, 83, 107) or rgb == (139, 69, 19) or rgb == (160, 82, 45) or rgb == (
            188, 143, 143)) or rgb[0] > 204 and rgb[1] > 104 and rgb[2] < 144:
                #print('Brown')
                return 'Brown'

            elif rgb[0] > 204 and rgb[1] < 193 and rgb[2] > 91:
                #print('Pink')
                return 'Pink'

            elif rgb == (255, 140, 0) or rgb == (255, 165, 0):
                #print('Orange')
                return 'Orange'

            elif rgb == (255, 215, 0):
                #print('Gold')
                return 'Gold'
            elif rgb == (189, 83, 107):
                #print('Green')
                return 'Green'
            else:
                #print('Red')
                return 'Red'

        elif rgb[1] > rgb[0] and rgb[1] > rgb[2] or rgb == (47, 79, 79):
            if rgb == (133, 130, 111) or rgb == (124, 125, 111):
                #print('Beige')
                return 'Beige'
            #print('green')
            return 'Green'

        elif rgb[2] > rgb[1] and rgb[2] > rgb[0] or rgb == (0, 255, 255) or rgb == (0, 139, 139) or rgb == (0, 128, 128):
            if rgb[0] > 122 and rgb[1] < 113 and rgb[2] > 203 or rgb == (128, 0, 128) or rgb == (75, 0, 130):
                #print('Purple')
                return 'Purple'
            else:
                #print('Blue')
                return 'Blue'

        elif rgb == (128, 128, 0):
            #print('Green')
            return 'Green'
        elif rgb == (255, 255, 0):
            #print('Yellow')
            return 'Yellow'
        elif rgb == (255, 0, 255) or rgb == (238, 130, 238) or rgb == (218, 112, 214) or rgb == (221, 160, 221):
            #print('Pink')
            return 'Pink'


    def features(self,distance):
        
        self.pose_points(self.frame)#Find pose points and detach body in three parts
        
        body_colors = []

        for parts in glob.glob('images/*'):

            if parts == 'images/cabeca.png':
                mask = ifmask('images/cabeca.png')
                print(mask)

            # Get the main color of the image
            output_color = self.color(parts)
            # Add to the body_colors list
            # First element[0] is the torso color, second[1] is the legs color
            body_colors.append(output_color)
        # Return the list of colors
        print(body_colors)
        height_estimate(distance, starty)
        return body_colors
cam = cv2.VideoCapture(2)   
print('start cam')
time.sleep(4)

if __name__ == "__main__":
    print('init code')
    __, frame = cam.read()
    cv2.imshow("frame",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):    
        cv2.destroyAllWindows()
    principal(frame)

    





