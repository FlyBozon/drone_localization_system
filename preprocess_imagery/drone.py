#main logic- i have good images from UAVID dataset, i can use them as calibration, i also created bad images based on them 
# (look in preprocess_for_uavid_training.py or effects.py - some distortion etc), and based on those imagery i can test if my preprocess functions work anyhow

import cv2
import numpy as np
import glob
import os
import shutil

# source_folder = "output/dataset/uavid/uavid_train/Images"
# distortion_folder = "dat_processing/distortion"
# calibration_folder = "dat_processing/calibration"

# os.makedirs(distortion_folder, exist_ok=True)
# os.makedirs(calibration_folder, exist_ok=True)

# for filename in os.listdir(source_folder):
#     source_path = os.path.join(source_folder, filename)

#     if not os.path.isfile(source_path):
#         continue

#     if "bad" in filename:
#         target_path = os.path.join(distortion_folder, filename)
#     else:
#         target_path = os.path.join(calibration_folder, filename)

#     shutil.copy2(source_path, target_path)
#     print(f"Copied: {filename} → {target_path}")


import numpy as np
import glob

pattern_size = (9, 6)

objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

objpoints = []
imgpoints = []

images = glob.glob("dat_processing/calibration/*.png")

print("Found", len(images), "PNG files\n")


for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size)

    if ret:
        cv2.drawChessboardCorners(img, pattern_size, corners, ret)

    cv2.imshow("corners", img)
    cv2.waitKey(500)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
