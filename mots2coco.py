import sys
if "/opt/ros/kinetic/lib/python2.7/dist-packages" in sys.path:
    sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")

import cv2
import numpy as np
import shutil
import os, glob
from shutil import copyfile
import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools
 
ROOT_DIR = '/home/kun/KITTI_MOTS/'
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
ANNOTATION_DIR = os.path.join(ROOT_DIR, "instances")
ANNOTATION_SAVE_DIR = os.path.join(ROOT_DIR, "annotations")
INSTANCE_DIR = os.path.join(ROOT_DIR, "instance_dir") 
IMAGE_SAVE_DIR = os.path.join(ROOT_DIR, "val_images")

INFO = {
    "description": "KITTI MOTS Training Dataset",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": "2020",
    "contributor": "Kevin_Jia",
    "date_created": "2020-3-21 19:19:19.123456"
}
 
LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'car',
        'supercategory': 'kitti mots',
    },
        {
        'id': 2,
        'name': 'pedestrian',
        'supercategory': 'kitti mots',
    }
]

background_label = [0, 10000]
idx=0
CROP = False
pic_scale = 1.0
h_bias = 1.0

def image_trans():
    img_subfolders = os.listdir(IMAGE_DIR)
    image_count = 0
    for sub in img_subfolders:
        # sub_path = sub + '/' + sub
        image_sub_path = os.path.join(IMAGE_DIR, sub)
        for image in os.listdir(image_sub_path):
            img_path = os.path.join(image_sub_path, image)
            ann_sub_path = os.path.join(ANNOTATION_DIR, sub)
            ann_path = os.path.join(ann_sub_path, image)
            if os.path.exists(ann_path): 
                img_save_path = os.path.join(IMAGE_SAVE_DIR, sub +"_"+image)
                ann_save_path = os.path.join(ANNOTATION_SAVE_DIR, sub +"_"+image)
                if CROP:
                    pic = cv2.imread(img_path)
                    h, w = pic.shape[:2]
                    new_w = w * pic_scale
                    new_h = new_w * (h/w)
                    top = int((h_bias*h-new_h)/2)
                    bottom = int((h_bias*h+new_h)/2)
                    left = int((w-new_w)/2)
                    right = int((w+new_w)/2)
                    roi = pic[top:bottom, left:right]
                    cv2.imwrite(img_save_path, roi) 
                    annotation = cv2.imread(ann_path, -1)
                    ann_roi = annotation[top:bottom, left:right]
                    cv2.imwrite(ann_save_path, ann_roi)
                else:
                    shutil.copy(img_path, img_save_path)
                    shutil.copy(ann_path, ann_save_path)
            else:
                print(image + '  do not have instance annotation')
            print(image_count)
            image_count += 1

def data_loader():
    imgs = os.listdir(IMAGE_SAVE_DIR)
    masks_generator(imgs, ANNOTATION_SAVE_DIR)

def masks_generator(images, ann_path):
    global idx
    pic_count = 0
    for pic_name in images:
        image_name = pic_name.split('.')[0]
        ann_folder = os.path.join(INSTANCE_DIR, image_name)
        os.mkdir(ann_folder)
        annotation_name = pic_name
        # print(annotation_name)
        annotation = cv2.imread(os.path.join(ann_path, annotation_name), -1)
        h, w = annotation.shape[:2]
        ids = np.unique(annotation)
        for id in ids:
            if id in background_label:
                continue
            else:
                class_id = id // 1000
                if class_id == 1:
                    instance_class = 'car'
                elif class_id == 2:
                    instance_class = 'pedestrian' 
                else:
                    continue    
            instance_mask = np.zeros((h, w, 3),dtype=np.uint8)
            mask = annotation == id
            instance_mask[mask] = 255
            mask_name = image_name + '_' + instance_class + '_' + str(idx) + '.png'
            cv2.imwrite(os.path.join(ann_folder, mask_name), instance_mask)
            idx += 1
        pic_count += 1
        print(pic_count)
 
def json_generate():
    car = 0
    pedestrian = 0
    files = os.listdir(IMAGE_SAVE_DIR)

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1

    # go through each image
    for image_filename in files:
        image_name = image_filename.split('.')[0]
        image_path = os.path.join(IMAGE_SAVE_DIR, image_filename)
        image = Image.open(image_path)
        image_info = pycococreatortools.create_image_info(
                image_id, os.path.basename(image_filename), image.size)
        coco_output["images"].append(image_info)
        print(image_filename)
        annotation_sub_path = os.path.join(INSTANCE_DIR, image_name)
        ann_files = os.listdir(annotation_sub_path)
        if len(ann_files) == 0:
            print("ao avaliable annotation")
            continue
        else:
            for annotation_filename in ann_files:
                annotation_path = os.path.join(annotation_sub_path, annotation_filename)
                for x in CATEGORIES:
                    if x['name'] in annotation_filename:
                        class_id = x['id']
                        break
                if class_id == 1:
                    car += 1
                elif class_id == 2:
                    pedestrian += 1
                else:
                    print('illegal class id')
                category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
                binary_mask = np.asarray(Image.open(annotation_path)
                                            .convert('1')).astype(np.uint8)

                annotation_info = pycococreatortools.create_annotation_info(
                        segmentation_id, image_id, category_info, binary_mask,
                        image.size, tolerance=2)

                if annotation_info is not None:
                    coco_output["annotations"].append(annotation_info)

                    segmentation_id = segmentation_id + 1

            image_id = image_id + 1
            print(image_id)
 
    with open('{}/val.json'.format(ROOT_DIR), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)
    print(car, pedestrian)

def data_normalize(data_root):
    # data_root = '/home/kun/KITTI_MOTS'
    train_path = os.path.join(data_root, "train/train_images")
    test_path = os.path.join(data_root, "val/val_images")
    # query_path = os.path.join(data_root, "query") 
    pic_list = []
    for pic in os.listdir(train_path):
        pic_list.append(os.path.join(train_path, pic))
    for pic in os.listdir(test_path):
        pic_list.append(os.path.join(test_path, pic)) 
    # for pic in os.listdir(query_path):
    #     pic_list.append(os.path.join(query_path, pic)) 

    R_means = []
    G_means = []
    B_means = []
    R_stds = []
    G_stds = []
    B_stds = []
    for pic in pic_list:
    # for pic in os.listdir(path):
        im = cv2.imread(pic)
        im_R = im[:,:,2]/255  #opencv default format is BGR
        im_G = im[:,:,1]/255
        im_B = im[:,:,0]/255
        im_R_mean = np.mean(im_R)
        im_G_mean = np.mean(im_G)
        im_B_mean = np.mean(im_B)
        im_R_std = np.std(im_R)
        im_G_std = np.std(im_G)
        im_B_std = np.std(im_B)
        R_means.append(im_R_mean)
        G_means.append(im_G_mean)
        B_means.append(im_B_mean)
        R_stds.append(im_R_std)
        G_stds.append(im_G_std)
        B_stds.append(im_B_std)
    means = [R_means,G_means,B_means]
    stds = [R_stds,G_stds,B_stds]
    mean = [0,0,0]
    std = [0,0,0]
    mean[0] = np.mean(means[0])
    mean[1] = np.mean(means[1])
    mean[2] = np.mean(means[2])
    std[0] = np.mean(stds[0])
    std[1] = np.mean(stds[1])
    std[2] = np.mean(stds[2])
    print('RGB MEAN:\n[{},{},{}]'.format(mean[0],mean[1],mean[2]))
    print('RGB VARIANCE:\n[{},{},{}]'.format(std[0],std[1],std[2]))
 
 
if __name__ == "__main__":
    # image_trans()
    # data_loader()
    # json_generate()
    data_normalize(ROOT_DIR)