import os
import dlib
import imutils
import cv2

import math
import matplotlib.pyplot as plt
import numpy as np

from find_eyes import find_eyes_from_image

def find_mouth_curve_from_image(im_file):
    im = np.float32(cv2.imread(im_file, cv2.IMREAD_GRAYSCALE) / 255.0)
    im_crop = im[20:, 100:260] 

    eye_list = find_eyes_from_image(im_crop)

    im = cv2.imread(im_file, cv2.IMREAD_GRAYSCALE)

    im_crop = im[20:, 100:260] 

    ret, thresh = cv2.threshold(im_crop, 60, 255, cv2.THRESH_BINARY)

    y_min = int(eye_list[0][1] + 0.19 * im_crop.shape[0])
    y_max = int(y_min + 0.19 * im_crop.shape[0])

    x_min = int(eye_list[0][0] - 0.17 * im_crop.shape[1])
    x_max = int(eye_list[1][0] + 0.17 * im_crop.shape[1])

    thresh = thresh[y_min:y_max, x_min:x_max]

    features = cv2.SIFT_create() 
      
    keypoints = features.detect(thresh, None)
    
    kp_len = len(keypoints)
    
    xcoords = np.zeros(kp_len)
    ycoords = np.zeros(kp_len)
    
    for i in range(kp_len):
        xcoords[i] = keypoints[i].pt[0]
        ycoords[i] = keypoints[i].pt[1]
    
    coeffs = np.polyfit(xcoords, ycoords, 2)
    
    # the first coefficient determines if the curve is up or down
    # the axes are flipped so multiply by -1
    result = coeffs[0] * -1

    # for seeing image with plotted points
    # output_image = cv2.drawKeypoints(thresh, keypoints, 0, (255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 

    if (result > 0): 
        return "happiness"
    elif (result < -0.001):
        return "sadness"
    else:
        return "neutral"