#!/usr/bin/env python3
from PIL import Image
from transformers import YolosFeatureExtractor, YolosForObjectDetection
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, ToPILImage
from colorthief import ColorThief
import webcolors
import glob
import cv2
import dlib
import numpy as np
import torch.utils.checkpoint as cp
import os
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image as Im
import time
from features_pkg.srv import Features
import rospkg


cats = ['shirt, blouse', 'top, t-shirt, sweatshirt', 'sweater', 'cardigan', 'jacket', 'vest', 'pants', 'shorts', 'skirt', 'coat', 'dress', 'jumpsuit', 'cape', 'glasses', 'hat', 'hair accessory', 'tie', 'glove', 'watch', 'belt', 'leg warmer', 'tights, stockings', 'sock', 'shoe', 'bag, wallet', 'scarf', 'umbrella', 'hood', 'collar', 'lapel', 'epaulette', 'sleeve', 'pocket', 'neckline', 'buckle', 'zipper', 'applique', 'bead', 'bow', 'flower', 'fringe', 'ribbon', 'rivet', 'ruffle', 'sequin', 'tassel']
COLORS =[[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],[0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
rospack =rospkg.RosPack()
directory = rospack.get_path('features_pkg')
directory = directory + '/src/'


class features():

    def __init__(self):
        rospy.Service('feature_detection', Features, self.handler)

        self.bridge = CvBridge()
        self.topic = '/zed_node/left_raw/image_raw_color' #/usb_cam/image_raw
        self.rate=rospy.Rate(5)
        self._check_cam_ready()
        self.img_sub = rospy.Subscriber(self.topic, Im, self.camera_callback)

    def _check_cam_ready(self):
      self.cam_img = None
      while self.cam_img is None and not rospy.is_shutdown():
         try:
               self.cam_img = rospy.wait_for_message(self.topic, Im, timeout=1.0)
               rospy.logdebug("Current "+self.topic+" READY=>" + str(self.cam_img))
         except:
               rospy.logerr("Current "+self.topic+" not ready yet, retrying.")

    def camera_callback(self,data):
        self.cam_img = data
        

    def ifglasses(self,path):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(directory+'shape_predictor_68_face_landmarks.dat')
        img = dlib.load_rgb_image(path)
        rect = detector(img,1)
        sp = predictor(img, rect[0])
        landmarks = np.array([[p.x, p.y] for p in sp.parts()])
        nose_bridge_x = []
        nose_bridge_y = []
        for i in [28,29,30,31,33,34,35]:
            nose_bridge_x.append(landmarks[i][0])
            nose_bridge_y.append(landmarks[i][1])

        x_min = min(nose_bridge_x)
        x_max = max(nose_bridge_x)
        y_min = landmarks[20][1]
        y_max = landmarks[31][1]
        img2 = Image.open(path)
        img2 = img2.crop((x_min,y_min,x_max,y_max))

        img_blur = cv2.GaussianBlur(np.array(img2),(3,3), sigmaX=0, sigmaY=0)
        edges = cv2.Canny(image =img_blur, threshold1=100, threshold2=200)
        
        edges_center = edges.T[(int(len(edges.T)/2))]

        if 255 in edges_center:
            return 'Glasses'
        else:
            return 'No glasses'


    def fix_channels(self,t):
        if len(t.shape) == 2:
            return ToPILImage()(torch.stack([t for i in (0, 0, 0)]))
        if t.shape[0] == 4:
            return ToPILImage()(t[:3])
        if t.shape[0] == 1:
            return ToPILImage()(torch.stack([t[0] for i in (0, 0, 0)]))
        return ToPILImage()(t)
    

    def idx_to_text(self,i):
        return str(cats[i])


    def box_cxcywh_to_xyxy(self,x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
            (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)


    def rescale_bboxes(self,out_bbox, size):
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b


    def plot_results(self,pil_img, prob, boxes):
        way = glob.glob(directory+'results/*')
        for py_file in way:
            try:
                os.remove(py_file)
            except OSError as e:
                rospy.logerr(f"Error:{ e.strerror}")
        detec = []
        color = COLORS * 100
        for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), color):
            cl = p.argmax()
            if cl<29 and ('jumpsuit' not in self.idx_to_text(cl)):
                if 'pants' in self.idx_to_text(cl):
                    xmin=(xmax+xmin)/2
                img = pil_img.crop((xmin+10,ymin+10,xmax-10,ymax-10))
                img.save(directory+'results/'+self.idx_to_text(cl)+'.png')
                detec.append([self.idx_to_text(cl),[int(xmin),int(ymax),int(xmax),int(ymin)]])


    def visualize_predictions(self,image, outputs, threshold=0.8):
        probas = outputs.logits.softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > threshold

        bboxes_scaled = self.rescale_bboxes(outputs.pred_boxes[0, keep].cpu(), image.size)

        self.plot_results(image, probas[keep], bboxes_scaled)
        

    def colorName(self,path):
        color_thief = ColorThief(path)
        dominant_color = color_thief.get_color()
        closest_color = None
        min_distance = float('inf')

        for color_name, rgb in webcolors.CSS3_NAMES_TO_HEX.items():
            css3_rgb = webcolors.hex_to_rgb(rgb)

            distance =self.delta_e_cie76(dominant_color, css3_rgb)

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


    def run_model(self,inputs):
        outputs = self.model(**inputs)
        return outputs
    

    def inferencing(self,path):
        MODEL_NAME = "valentinafeve/yolos-fashionpedia"

        feature_extractor = YolosFeatureExtractor.from_pretrained('hustvl/yolos-small')
        self.model = YolosForObjectDetection.from_pretrained(MODEL_NAME)

        image = Image.open(open(path, "rb"))
        image = self.fix_channels(ToTensor()(image))
        image = image.resize((600, 800))
        inputs = feature_extractor(images=image, return_tensors="pt")
        outputs = cp.checkpoint(self.run_model, inputs)
        self.visualize_predictions(image, outputs, threshold=0.5)


    def main(self):
    
        frame = self.bridge.imgmsg_to_cv2(self.cam_img, desired_encoding='bgr8')
        time.sleep(1)
       
        cv2.imwrite(directory+'base/img.png', frame)
        path = directory+'base/img.png'

        self.inferencing(path)
        out = ''
        for clothes in glob.glob(directory+'results/*'):
            color = self.colorName(clothes)
            name = clothes.split('results/')[1].split('.')[0]
            out += str(name + ' ' + color + '/')

        if directory+'results/shoe.png' not in glob.glob(directory+'results/*'):
            out += 'No shoe/'

        if directory+'results/hat.png' not in glob.glob(directory+'results/*') and directory+'results/cap.png' not in glob.glob(directory+'results/*') \
        and directory+'results/hair accessory.png' not in glob.glob(directory+'results/*'):
            out += 'No head accessory/'

        out = out + self.ifglasses(directory+"base/img.png")
        return out
    

    def handler(self, request):
        self.recog = 0
        rospy.loginfo("Service called!")
        rospy.loginfo("Requested..")

        time.sleep(3)
        
        while self.recog == 0:
            self.img_sub = rospy.Subscriber(self.topic,Im,self.camera_callback)
            output = self.main()
            self.rate.sleep()
            return output
        cv2.destroyAllWindows()


if __name__ == '__main__':
    rospy.init_node('feature_detection')
    rospy.loginfo("Service started!")

    features()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")   



