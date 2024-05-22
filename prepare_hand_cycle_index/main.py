#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/code/prepare_hand_cycle_index/main.py
Project: /workspace/code/prepare_hand_cycle_index
Created Date: Wednesday May 22nd 2024
Author: Kaixu Chen
-----
Comment:
Use mediapipe's hand marker to find boundary values in the hand motion cycle from the video
Have a good code time :)
-----
Last Modified: Wednesday May 22nd 2024 3:25:33 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2024 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------

22-05-2024	Kaixu Chen	initial version
'''

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2 
import hydra
import math

from pathlib import Path
from torchvision.io import read_video
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import os
os.environ['MEDIAPIPE_GPU_VERSION'] = '1'  # 选择CUDA计算能力，例如2表示使用CUDA Compute Capability 2.x的设备


MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

# def draw_landmarks_on_image(rgb_image, detection_result):
#   hand_landmarks_list = detection_result.hand_landmarks
#   handedness_list = detection_result.handedness
#   annotated_image = np.copy(rgb_image)

#   # Loop through the detected hands to visualize.
#   for idx in range(len(hand_landmarks_list)):
#     hand_landmarks = hand_landmarks_list[idx]
#     handedness = handedness_list[idx]

#     # Draw the hand landmarks.
#     hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
#     hand_landmarks_proto.landmark.extend([
#       landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
#     ])
#     solutions.drawing_utils.draw_landmarks(
#       annotated_image,
#       hand_landmarks_proto,
#       solutions.hands.HAND_CONNECTIONS,
#       solutions.drawing_styles.get_default_hand_landmarks_style(),
#       solutions.drawing_styles.get_default_hand_connections_style())

#     # Get the top left corner of the detected hand's bounding box.
#     height, width, _ = annotated_image.shape
#     x_coordinates = [landmark.x for landmark in hand_landmarks]
#     y_coordinates = [landmark.y for landmark in hand_landmarks]
#     text_x = int(min(x_coordinates) * width)
#     text_y = int(min(y_coordinates) * height) - MARGIN

#     # Draw handedness (left or right hand) on the image.
#     cv2.putText(annotated_image, f"{handedness[0].category_name}",
#                 (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
#                 FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

#   return annotated_image

def calc_distance(hand_landmarks):
    middle_finger_tip = hand_landmarks[0][12]
    wrist = hand_landmarks[0][0]

    distance = np.sqrt((middle_finger_tip.x - wrist.x) ** 2 + (middle_finger_tip.y - wrist.y) ** 2)
    return distance

def process_one_video(frames, detector):
    
    dis_res = [] 

    for img in frames:
        # STEP 3: Load the input image.
        # image = mp.Image.create_from_numpy_array(frames)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img.numpy())

        # STEP 4: Detect hand landmarks from the input image.
        results = detector.detect(image)

        handedness = results.handedness
        hand_landmarks = results.hand_landmarks

        dis_res.append(calc_distance(hand_landmarks))

    return dis_res

def process_label(label: Path, detector):
    for one_person in label.iterdir():
        for one_video in one_person.iterdir():
            vframes, _, info = read_video(one_video)
            dis_res = process_one_video(vframes, detector)

            # TODO: 从这里判断张开和闭合的界限
            max_dis = max(dis_res)
            min_dis = min(dis_res)


@hydra.main(config_path="/workspace/code/config/", config_name="prepare_dataset")
def main(cfg) -> None:

    raw_data_path = Path(cfg.extract_dataset.data_path)    
    save_data_path = Path(cfg.extract_dataset.save_path) 
    ckpt = cfg.mediapipe.hand_ckpt

    base_options = python.BaseOptions(model_asset_path=ckpt)
    options = vision.HandLandmarkerOptions(base_options=base_options,
                                        num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)

    for label in raw_data_path.iterdir():
       process_label(label, detector)

if '__main__' == __name__:
    main()