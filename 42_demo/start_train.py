#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics-8.2.77 
@File    ：start_train.py
@IDE     ：PyCharm 
@Author  ：肆十二（付费咨询QQ: 3045834499） 粉丝可享受99元调试服务
@Description  ：TODO 添加文件描述
@Date    ：2024/8/15 15:14 
'''
import time
from ultralytics import YOLO

# Load a model 测试添加注释
# yolov8n模型训练
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
results = model.train(data='A_my_data.yaml', epochs=100, imgsz=640, device=[0,], workers=0, batch=4, cache=True)
time.sleep(10)