#!/usr/bin/env python3
import tensorflow as tf
import cv2
import os
import re
import PIL.Image as Image
import numpy as np

# Function check_file() is to get the aiming file list in the given folder 
# folder can be set to 2 arguments: img and ann
# image file type is png, annotation file types can be png and txt 


class Data_Viewer:
    def __init__(self, img_folder, ann_folder):
        self._img_folder = img_folder
        self._ann_folder = ann_folder
        self.show_img = True
        self.show_fusion = True
        self.save_imgvideo = False
        self.save_fusedvideo = False

        self._car_color = [[25, 25, 112],[100, 149, 237],[106, 90, 205],[65, 105, 225],[135, 206, 235],
                            [176, 224, 230],[0, 255, 255],[46, 139, 87],[32, 178, 170],[0, 255, 127],
                            [107, 142, 35],[34, 139, 34],[50, 205, 50],[123, 104, 238],[95, 158, 160],
                            [0, 100, 0],[112, 128, 144],[175, 238, 238],[0, 250, 154],[47, 79, 79]]

        self._ped_color = [[255, 255, 0],[238, 221, 130],[188, 143, 143],[205, 92, 92],[139, 69, 19],[205, 133, 63],
                            [244, 164, 96],[210, 105, 30],[178, 34, 34],[250, 128, 114],[255, 165, 0],[255, 69, 0],
                            [255, 105, 180,],[176, 48, 96],[255, 0, 255],[218, 112, 214],[139, 69, 19],[255, 99, 71],
                            [219, 112, 147],[153, 50, 204]]

        self.imglist = []
        self.annlist = []
        self.pic_size = ()
        self.color_mask = []
        

    def check_file(self, folder, filetype):
        if folder == 'img':
            folder_path = self._img_folder
        elif folder == 'ann':
            folder_path = self._ann_folder
        else:
            print("***********illegal folder************") 
        files = os.listdir(folder_path)
        #print(files)
        if filetype == 'png':
            aimfile = re.compile(r'.*png')
            suffix = '.png'
        elif filetype == 'txt':
            aimfile = re.compile(r'.*txt')
            suffix = '.txt'
        else:
            print("***********illegal type**************")    
        num = []
        output = []
        for i in files:
            if re.match(aimfile,i):
                num.append(int(i.split('.')[0]))
        num.sort()
        for n in num:
            output.append(folder_path + str("{:0>6d}".format(n)) + suffix)
        #print(output)
        if folder == 'img':
            self.imglist = output
            pic0 = cv2.imread(self.imglist[0])
            self.pic_size = (pic0.shape[0],pic0.shape[1])
        else:
            self.annlist = output

    def visualize_data(self):
        if self.show_img:

            if self.save_imgvideo:
                img_video = cv2.VideoWriter()
                img_video.open(self._img_folder+'images.avi', cv2.VideoWriter_fourcc("I", "4", "2", "0"), 10, (self.pic_size[1], self.pic_size[0]))
            if self.save_fusedvideo:
                fused_video = cv2.VideoWriter()
                fused_video.open(self._img_folder+'instances.avi', cv2.VideoWriter_fourcc("I", "4", "2", "0"), 10, (self.pic_size[1], self.pic_size[0]))
                
            for i in range(len(self.imglist)):
                pic = cv2.imread(self.imglist[i])
                if self.show_fusion:
                    self.color_mask = np.zeros([self.pic_size[0], self.pic_size[1] , 3], np.uint8)
                    anno = cv2.imread(self.annlist[i], -1)    #Convert the single channel uint16 image to uint8 gray scale image
                              
                    obj_ids = np.unique(anno)
                    for id in obj_ids:
                        if id == 0:
                            pass
                        elif id // 1000 == 1:
                            instance_id = id % 1000
                            for point in np.argwhere(anno == id):
                                self.color_mask[point[0]][point[1]] = self._car_color[instance_id % 20]
                        elif id // 1000 == 2:
                            instance_id = id % 1000
                            for point in np.argwhere(anno == id):
                                self.color_mask[point[0]][point[1]] = self._ped_color[instance_id % 20]
                        else:
                            for point in np.argwhere(anno == id):
                                self.color_mask[point[0]][point[1]] = [255, 255, 255]

                    fused_pic = cv2.addWeighted(pic, 0.6, self.color_mask, 0.4, 1)
                    #cv2.imshow('Mask', self.color_mask)
                    cv2.imshow('ANNOTATION', fused_pic)
                cv2.imshow('VIDEO', pic)

                if self.save_imgvideo:
                    img_video.write(pic)
                if self.save_fusedvideo:
                    fused_video.write(fused_pic)
                cv2.waitKey(1)

            if self.save_imgvideo:
                img_video.release()
            if self.save_fusedvideo:
                fused_video.release()


if __name__ == '__main__':
    img_folder = '/media/data/KITTI_MOTS/train/images/0001/'
    ann_folder = '/media/data/KITTI_MOTS/train/instances/0001/'
    data_viewer = Data_Viewer(img_folder, ann_folder)
    data_viewer.check_file('img', 'png')
    data_viewer.check_file('ann', 'png')
    data_viewer.visualize_data()
    cv2.destroyAllWindows()
