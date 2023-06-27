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

# This is the order of the categories list. NO NOT CHANGE. Just for visualization purposes
IMAGE_PATH = "base/img2.jpg"
cats = ['shirt, blouse', 'top, t-shirt, sweatshirt', 'sweater', 'cardigan', 'jacket', 'vest', 'pants', 'shorts', 'skirt', 'coat', 'dress', 'jumpsuit', 'cape', 'glasses', 'hat', 'headband, head covering, hair accessory', 'tie', 'glove', 'watch', 'belt', 'leg warmer', 'tights, stockings', 'sock', 'shoe', 'bag, wallet', 'scarf', 'umbrella', 'hood', 'collar', 'lapel', 'epaulette', 'sleeve', 'pocket', 'neckline', 'buckle', 'zipper', 'applique', 'bead', 'bow', 'flower', 'fringe', 'ribbon', 'rivet', 'ruffle', 'sequin', 'tassel']

def ifglasses(path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    img = dlib.load_rgb_image(path)
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
    
def fix_channels(t):
    if len(t.shape) == 2:
        return ToPILImage()(torch.stack([t for i in (0, 0, 0)]))
    if t.shape[0] == 4:
        return ToPILImage()(t[:3])
    if t.shape[0] == 1:
        return ToPILImage()(torch.stack([t[0] for i in (0, 0, 0)]))
    return ToPILImage()(t)
def idx_to_text(i):
    return str(cats[i])

# Random colors used for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def plot_results(pil_img, prob, boxes):
    way = glob.glob('results/*')
    for py_file in way:
        try:
            os.remove(py_file)
        except OSError as e:
            rospy.logerr(f"Error:{ e.strerror}")
    detec = []
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        cl = p.argmax()
        if cl<29 and ('jumpsuit' not in idx_to_text(cl)):
          if 'pants' in idx_to_text(cl):
            xmin=(xmax+xmin)/2
          img = pil_img.crop((xmin+10,ymin+10,xmax-10,ymax-10))
          img.save('results/'+idx_to_text(cl)+'.png')
          detec.append([idx_to_text(cl),[int(xmin),int(ymax),int(xmax),int(ymin)]])
    return detec

def visualize_predictions(image, outputs, threshold=0.8):
    # keep only predictions with confidence >= threshold
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold

    # convert predicted boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs.pred_boxes[0, keep].cpu(), image.size)


    # plot results
    return plot_results(image, probas[keep], bboxes_scaled)

def colorName(path):
    color_thief = ColorThief(path)

    dominant_color = color_thief.get_color()
    closest_color = None
    min_distance = float('inf')
    for color_name, rgb in webcolors.CSS3_NAMES_TO_HEX.items():
        css3_rgb = webcolors.hex_to_rgb(rgb)

        distance =delta_e_cie76(dominant_color, css3_rgb)

        if distance < min_distance:
            min_distance = distance
            closest_color = color_name
    return closest_color

def delta_e_cie76(color1, color2):
    r_mean = (color1[0] + color2[0]) / 2
    delta_r = color1[0] - color2[0]
    delta_g = color1[1] - color2[1]
    delta_b = color1[2] - color2[2]
    return ((2 + (r_mean / 256)) * delta_r ** 2 + 4 * delta_g ** 2 + (2 + ((255 - r_mean) / 256)) * delta_b ** 2) ** 0.5

def run_model(inputs):
    outputs = model(**inputs)
    return outputs

MODEL_NAME = "valentinafeve/yolos-fashionpedia"
feature_extractor = YolosFeatureExtractor.from_pretrained('hustvl/yolos-small')
model = YolosForObjectDetection.from_pretrained(MODEL_NAME)
image = Image.open(open(IMAGE_PATH, "rb"))
image = fix_channels(ToTensor()(image))
image = image.resize((600, 800))

inputs = feature_extractor(images=image, return_tensors="pt")
outputs = cp.checkpoint(run_model, inputs)
detec = visualize_predictions(image, outputs, threshold=0.5)

body_colors =[]
out = ''
for clothes in glob.glob('results/*'):
  color = colorName(clothes)
  name = clothes.split('/')[1].split('.')[0]
  out +=str(name+' '+color+'/')
if 'shoe' not in glob.glob('results/*'):
  out+='No shoe'
out = out+'/'+ifglasses(IMAGE_PATH)
print(out)



