import sys
if "/opt/ros/kinetic/lib/python2.7/dist-packages" in sys.path:
    sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
import os
import cv2
import numpy as np
import shutil

source = 'mask'
output = "mask2"

imgs = os.listdir(source)
for img in imgs:
    path = os.path.join(source, img)
    pic = cv2.imread(path)
    resized = cv2.resize(pic, (128,128))
    out_path = os.path.join(output, img)
    cv2.imwrite(out_path, resized)