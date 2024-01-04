

# USe colab to run code
from google.colab import drive
drive.mount('/content/drive')

!pip install ultralytics==8.0.54

!yolo task=detect mode=train epochs=100 data=/content/drive/MyDrive/Website_Detection/Dataset3/data.yaml model=yolov8m.pt imgsz=448 batch=8

from ultralytics import YOLO
import torch
model = YOLO("/content/runs/detect/train/weights/last.pt","v8")
source ="/content/drive/MyDrive/Website_Detection/Splash Screen.png"

output = model.predict(source,save=True,hide_labels = False,line_thickness=1)

