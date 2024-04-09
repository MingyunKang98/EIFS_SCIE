from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load a pretrained YOLOv8n model
model = YOLO("./runs/segment/train6/weights/best.pt")

# Run inference on an image

dir = "../ultralytics/datasets/train/images/IMG_4245_JPG.rf.63460a44ffb7591e4ab7aaa7483d11e6.jpg"
results = model(dir, save=False, save_txt=True,save_crop=True,show_conf = False,show_labels=False,show_boxes=False)  # list of 1 Results object

