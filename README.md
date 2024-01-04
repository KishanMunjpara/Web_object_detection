# Web Element Detection using YOLOv8

## Overview

This project implements a computer vision model using YOLOv8 for detecting and identifying various web elements in Figma webpages. The model can identify elements such as 'button', 'field', 'heading', 'iframe', 'image', 'label', 'link', 'text'. It creates bounding boxes around each identified element and tags them with the corresponding element name.

### Prerequisites

- Python 3.x
- YOLOv8 (ultralytics 8.0.54)
- torch

## DATASET
Dataset:![dataset3.zip](https://drive.google.com/file/d/1jaQZC17SsnYpR9D7yhpsKmacUvbvLTgt/view?usp=drive_link)
## CODE

CODE FILE :
1) PYTHON :[web_object_detection.py](web_object_detection.py).
2) Jupyter Notebook :[web_object_detection.ipynb](Web_Object_detection.ipynb).

## Sample Reference Images

Sample reference images for evaluation can be found [here](https://www.figma.com/community/file/1132396044075007632/tortilicious-a-fast-food-app).

## Sample Results

Sample results generated by the model can be viewed [here](https://drive.google.com/drive/folders/1TzavbXxacf8e9a4z_TvDkfn8RDlZAqZr?usp=drive_link).

![Sample Result](https://drive.google.com/uc?id=17HFr533KcYwnzmsV3Yw9OJEuEfWmzGSB)

## Weights

There are Two weight file
1) Best Point :![Best.pt](https://drive.google.com/file/d/1Fa9gQ_0QcIxL7XwDx1YGSzxrOR-eM4sG/view?usp=drive_link)
2) lats point:![last.pt](https://drive.google.com/file/d/1NJtglgMWsC3_yLIBQvHE5BsYXVm3S7Vi/view?usp=drive_link)

## Evaluation
1) confusion Matrix : ![](https://drive.google.com/uc?id=1z-2FN5g1zu002-8kwyr1UOog_brGCR3W)
2) F1 curve :![](https://drive.google.com/uc?id=1gAcMpI4ScfMALOfSYtG-ZGE8D2Jgw-jr)
3) P curve :![](https://drive.google.com/uc?id=1rw0MkFhU6OOkY486wHv8O2E_zvja1OQW)
4) R curve :![](https://drive.google.com/uc?id=1x7Vyp34cFotqGkG9gmt3O1L-WLYnmIqt)
5) PR curve :![](https://drive.google.com/uc?id=1x8d9B54EnAfsprtAjYQi9M6JQWFBJFWw)


<img src="https://drive.google.com/uc?id=1z-2FN5g1zu002-8kwyr1UOog_brGCR3W" alt="Confusion Matrix" style="max-width: 100%; height: auto;">
<p float="left">
  <img src="https://drive.google.com/uc?id=1gAcMpI4ScfMALOfSYtG-ZGE8D2Jgw-jr" alt="F1 Curve" style="width: 45%;padding: 5px;">
  <img src="https://drive.google.com/uc?id=1rw0MkFhU6OOkY486wHv8O2E_zvja1OQW" alt="Precision Curve" style="width: 45%;padding: 5px;">
</p>
<p float="left">
  <img src="https://drive.google.com/uc?id=1x7Vyp34cFotqGkG9gmt3O1L-WLYnmIqt" alt="Recall Curve" style="width: 45%;padding: 5px;">
  <img src="https://drive.google.com/uc?id=1x8d9B54EnAfsprtAjYQi9M6JQWFBJFWw" alt="PR Curve" style="width: 45%;padding: 5px;">
</p>
