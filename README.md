##  Installation
- This project is a modified version based on yolo8 https://github.com/ultralytics/ultralytics 
- please check this link for more info

1. git clone
2. cd to your folder
3. conda create -n yolo python==3.8.5
4. conda activate yolo
5. conda install pytorch(based on your cuda version)
6. pip install -v -e .

##  Train/Inference
- use the srcipt in "wukong" folder for training and inference
- my base model is yolov8l

##  Model
- orginal yolov8l model:https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l.pt
- Wukong model:https://huggingface.co/archifancy/YOLO_for_WUKONG/tree/main
- Wukong dataset:https://huggingface.co/datasets/archifancy/YOLO-Wukong/tree/main



