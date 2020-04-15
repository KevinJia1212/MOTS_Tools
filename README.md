# MOTS_Tools
Tools for multi-object tracking and segmentation(MOTS) task.

# SOURCE
This code can show the images and annotations of the MOTS dataset, which is published by Prof.Dr.Bastian Leibe of RWTH-AACHEN university at https://www.vision.rwth-aachen.de/page/mots, and thier project link is https://github.com/VisualComputingInstitute/TrackR-CNN

# Data Generation
data_visualize.py: Edit the image sequence path and the corresponding annotation path at __main__ function, and the __init__ function of the class Data_Viewer provides 4 options for users to set whether you want to save the image video or annotation video.

mots2reid.py: Crop the instances of cars and pedetrians in the KITTI MOTS dataset to generate a person-vehicle re-identification dataset

mots2coco.py: Convert the KITTI MOTS dataset to a instance segmentation dataset in coco annotation style, and category includes car and pedestrian.

visualize_coco.py: Randomly show a picture with its annotation.

### Evaluating a tracking result
Clone this repository, navigate to the mots_tools directory and make sure it is in your Python path. 
Now suppose your tracking results are located in a folder "tracking_results". Suppose further the ground truth annotations are located in a folder "gt_folder". Then you can evaluate your results using the commands
```
python mots_eval/eval.py tracking_results gt_folder seqmap
```
where "seqmap" is a textfile containing the sequences which you want to evaluate on. Several seqmaps are already provided in the mots_eval repository: val.seqmap, train.seqmap, fulltrain.seqmap, val_MOTSchallenge.seqmap which correspond to the KITTI MOTS validation set, the KITTI MOTS training set, both KITTI MOTS sets combined and the four annotated MOTSChallenge sequences respectively.

Parts of the evaluation logic are built upon the KITTI 2D tracking evaluation devkit from http://www.cvlibs.net/datasets/kitti/eval_tracking.php

### Visualizing a tracking result
Similarly to evaluating tracking results, you can also create visualizations using
```
python mots_eval/visualize_mots.py tracking_results img_folder output_folder seqmap
```
where "img_folder" is a folder containing the original KITTI tracking images (http://www.cvlibs.net/download.php?file=data_tracking_image_2.zip) and "output_folder" is a folder where the resulting visualization will be created.
