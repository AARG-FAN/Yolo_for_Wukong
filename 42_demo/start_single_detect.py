#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics-8.2.77 
@File    ：start_single_detect.py
@IDE     ：PyCharm 
@Author  ：肆十二（付费咨询QQ: 3045834499） 粉丝可享受99元调试服务
@Description  ：TODO 添加文件描述
@Date    ：2024/8/15 15:15 
'''
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model(["images/resources/demo.jpg", ])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="images/resources/result.jpg")  # save to disk

"""推理可选参数
source	str	'ultralytics/assets'	Specifies the data source for inference. Can be an image path, video file, directory, URL, or device ID for live feeds. Supports a wide range of formats and sources, enabling flexible application across different types of input.
conf	float	0.25	Sets the minimum confidence threshold for detections. Objects detected with confidence below this threshold will be disregarded. Adjusting this value can help reduce false positives.
iou	float	0.7	Intersection Over Union (IoU) threshold for Non-Maximum Suppression (NMS). Lower values result in fewer detections by eliminating overlapping boxes, useful for reducing duplicates.
imgsz	int or tuple	640	Defines the image size for inference. Can be a single integer 640 for square resizing or a (height, width) tuple. Proper sizing can improve detection accuracy and processing speed.
half	bool	False	Enables half-precision (FP16) inference, which can speed up model inference on supported GPUs with minimal impact on accuracy.
device	str	None	Specifies the device for inference (e.g., cpu, cuda:0 or 0). Allows users to select between CPU, a specific GPU, or other compute devices for model execution.
max_det	int	300	Maximum number of detections allowed per image. Limits the total number of objects the model can detect in a single inference, preventing excessive outputs in dense scenes.
vid_stride	int	1	Frame stride for video inputs. Allows skipping frames in videos to speed up processing at the cost of temporal resolution. A value of 1 processes every frame, higher values skip frames.
stream_buffer	bool	False	Determines if all frames should be buffered when processing video streams (True), or if the model should return the most recent frame (False). Useful for real-time applications.
visualize	bool	False	Activates visualization of model features during inference, providing insights into what the model is "seeing". Useful for debugging and model interpretation.
augment	bool	False	Enables test-time augmentation (TTA) for predictions, potentially improving detection robustness at the cost of inference speed.
agnostic_nms	bool	False	Enables class-agnostic Non-Maximum Suppression (NMS), which merges overlapping boxes of different classes. Useful in multi-class detection scenarios where class overlap is common.
classes	list[int]	None	Filters predictions to a set of class IDs. Only detections belonging to the specified classes will be returned. Useful for focusing on relevant objects in multi-class detection tasks.
retina_masks	bool	False	Uses high-resolution segmentation masks if available in the model. This can enhance mask quality for segmentation tasks, providing finer detail.
embed	list[int]	None	Specifies the layers from which to extract feature vectors or embeddings. Useful for downstream tasks like clustering or similarity search.
"""

"""可视化可选参数
show	bool	False	If True, displays the annotated images or videos in a window. Useful for immediate visual feedback during development or testing.
save	bool	False	Enables saving of the annotated images or videos to file. Useful for documentation, further analysis, or sharing results.
save_frames	bool	False	When processing videos, saves individual frames as images. Useful for extracting specific frames or for detailed frame-by-frame analysis.
save_txt	bool	False	Saves detection results in a text file, following the format [class] [x_center] [y_center] [width] [height] [confidence]. Useful for integration with other analysis tools.
save_conf	bool	False	Includes confidence scores in the saved text files. Enhances the detail available for post-processing and analysis.
save_crop	bool	False	Saves cropped images of detections. Useful for dataset augmentation, analysis, or creating focused datasets for specific objects.
show_labels	bool	True	Displays labels for each detection in the visual output. Provides immediate understanding of detected objects.
show_conf	bool	True	Displays the confidence score for each detection alongside the label. Gives insight into the model's certainty for each detection.
show_boxes	bool	True	Draws bounding boxes around detected objects. Essential for visual identification and location of objects in images or video frames.
line_width	None or int	None	Specifies the line width of bounding boxes. If None, the line width is automatically adjusted based on the image size. Provides visual customization for clarity.
"""