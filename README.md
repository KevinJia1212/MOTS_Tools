# MOTS_Tools
Data visualization and evaluation tools for multi-object tracking and segmentation(MOTS) task.

This code can show the images and annotations of the MOTS dataset, which is published by Prof.Dr.Bastian Leibe of RWTH-AACHEN university at https://www.vision.rwth-aachen.de/page/mots, and thier project link is https://github.com/VisualComputingInstitute/TrackR-CNN

data_visualize.py: Edit the image sequence path and the corresponding annotation path at __main__ function, and the __init__ function of the class Data_Viewer provides 4 options for users to set whether you want to save the image video or annotation video.

mots2reid.py: Crop the instances of cars and pedetrians in the KITTI MOTS dataset to generate a person-vehicle re-identification dataset

mots2coco.py: Convert the KITTI MOTS dataset to a instance segmentation dataset in coco annotation style, and category includes car and pedestrian.

visualize_coco.py: Randomly show a picture with its annotation.