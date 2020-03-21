import cv2
import numpy as np
import os
from collections import defaultdict
import random
import shutil
import glob

INS_COUNT = 0
AREA_THRESH = pow(64,2)
TEST_ALPHA = 0.3
# reid pic name : InstanceId _ c1s1 _ FrameId _ SequenceId .jpg

root = "/home/kun/KITTI_MOTS/"
save_dir = os.path.join(root, "mask_reid")

def crop(img_path, ann_path, save_path, crop_mask=False):
    global INS_COUNT
    pic_count = 0
    seq_id = "_" + os.path.basename(img_path)
    print("Processing sequence%s"%seq_id)
    id_temp = 0

    imgs = []
    anns = []
    img_list = os.listdir(img_path)
    img_list.sort()
    # for pic in img_list:
    #     img = cv2.imread(os.path.join(img_path, pic))
    #     imgs.append(img)
    #     ann = cv2.imread(os.path.join(ann_path, pic), -1)
    #     anns.append(ann)
    id_map = {}
    for pic_name in img_list:
        print(pic_name)
        img = cv2.imread(os.path.join(img_path, pic_name))
        ann = cv2.imread(os.path.join(ann_path, pic_name), -1)
        ins_ids = np.unique(ann)
        for id in ins_ids:
            if id in [0, 10000]:
                continue
            else:
                mask = np.argwhere(ann==id)
                if len(mask) > AREA_THRESH:
                    if id in id_map.keys():
                        reid = id_map[id]
                    else:
                        reid = INS_COUNT
                        id_map[id] = reid
                        INS_COUNT += 1
                
                    v = mask[:, 0]
                    h = mask[:, 1]
                    top = v.min()
                    bottom = v.max()
                    left = h.min()
                    right = h.max()
                    roi = img[top:bottom, left:right]
                    if crop_mask:
                        mask_roi = ann[top:bottom, left:right]
                        nm = mask_roi != id
                        roi[nm] = 0 
                    reid_name = str(reid).zfill(6) + "_c1s1_" + pic_name.split('.')[0] + seq_id + ".jpg"
                    cv2.imwrite(os.path.join(save_path, reid_name), roi)


def data_normalize(data_root):
    # data_root = '/home/kun/KITTI_MOTS'
    train_path = os.path.join(data_root, "train")
    test_path = os.path.join(data_root, "test")
    query_path = os.path.join(data_root, "query") 
    pic_list = []
    for pic in os.listdir(train_path):
        pic_list.append(os.path.join(train_path, pic))
    for pic in os.listdir(test_path):
        pic_list.append(os.path.join(test_path, pic)) 
    for pic in os.listdir(query_path):
        pic_list.append(os.path.join(query_path, pic)) 

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

def dividing(source_dir, alpha):
    train_dir = os.path.join(source_dir, "train")
    test_dir = os.path.join(source_dir, "test")
    query_dir = os.path.join(source_dir, "query")
    os.mkdir(train_dir)
    os.mkdir(test_dir)
    os.mkdir(query_dir)
    ids = defaultdict(list)
    paths = glob.glob(os.path.join(source_dir, '*.jpg'))
    for path in paths:
        img = os.path.basename(path)
        id = img.split('_')[0]
        ids[id].append(img)
    train = []
    test = []
    query = []
    for id in ids:
        id_list = ids[id]
        length = len(id_list)
        if length >= 5:
            idxes = list(range(length))
            test_idx = random.sample(idxes, round(length*alpha))
            for idx in test_idx:
                idxes.remove(idx)
            query_idx = random.choice(test_idx)
            test_idx.remove(query_idx)
            query.append(id_list[query_idx])
            for idx in test_idx:
                test.append(id_list[idx])
            for idx in idxes:
                train.append(id_list[idx])
    for img in train:
        shutil.move(os.path.join(source_dir, img), os.path.join(train_dir, img))
    for img in test:
        shutil.move(os.path.join(source_dir, img), os.path.join(test_dir, img))
    for img in query:
        shutil.move(os.path.join(source_dir, img), os.path.join(query_dir, img))
# def show(path):
#     pics = os.listdir(path)
#     for pic in pics:
#         img = cv2.imread(os.path.join(path, pic))
#         cv2.imshow('pic', img)
#         cv2.waitKey(0)


if __name__ == "__main__":
    # if not os.path.exists(save_dir):
    #     os.mkdir(save_dir)  
    # sub_folders = ["train", "val"]
    # for sub in sub_folders:  
    #     sub_path = os.path.join(root, sub)
    #     image_dir = os.path.join(sub_path, "images")
    #     ann_dir = os.path.join(sub_path, "instances")
    #     for seq in os.listdir(image_dir):
    #         image_path = os.path.join(image_dir, seq)
    #         ann_path = os.path.join(ann_dir, seq)
    #         crop(image_path, ann_path, save_dir, crop_mask=True)
    # dividing(save_dir, TEST_ALPHA)
    data_normalize(save_dir)
    
    