# vedai-Yolov8
A python script to trail a YOLO model on Vedai dataset

# Requirements
```
pip install -r requirements.txt
```
# Train
## 1. Prepare training data
1.1 Download VEDAI dataset for our training from [VEDAI](https://downloads.greyc.fr/vedai/)

1.2 Unzip the dataset and arrange the files in provided order for transforming script to work :

```
├── dataset
│   ├── VEDAI
│   │   ├── images
│   │   ├── labels
│   │   ├── fold01.txt
│   │   ├── fold01test.txt
│   │   ├── fold02.txt
│   │   ├── .....
│   ├── VEDAI_1024
│   │   ├── images
│   │   ├── labels
```

1.3 Run the transform.py script to convert the annotation format from PascalVOC to YOLO Horizontal Boxes.

1.4 Classify the images in **train**, **val** and **test** with the following folder structure :
```
├── data
│   ├── train
│   │   ├── images
│   │   ├── labels
│   ├── val
│   │   ├── images
│   │   ├── labels
│   ├── test
│   │   ├── images
│   │   ├── labels
```

`Note : Adjust the path='dataset' before running the script.`
## 2. Begin the training using the CLI command :
2.1 Update data.yaml with the location of dataset
2.2 Run the following CLI command
```
yolo task=detect mode=train epochs=100 data=data.yaml model=yolov8m.pt imgsz=512 batch=8
```
# Test
1. The trained weight would be stored in runs/detect/train/weights/
2. Run [Detection.py](https://github.com/Nishantdd/vedai-Dotav8/blob/main/Detection.py) with the updated location of the weight

`Note : Update the location for video file`

# Detection results
## Sample Run
<p align='center'>
<img alt="Vedai" width="800" src="https://github.com/Nishantdd/vedai-Dotav8/blob/main/img/P205.png"></p>

## Pretrained vs Vedai
<p align='center'>
<img alt="Vedai" width="800" src="https://github.com/Nishantdd/vedai-Dotav8/blob/main/img/Compar.png"></p>
