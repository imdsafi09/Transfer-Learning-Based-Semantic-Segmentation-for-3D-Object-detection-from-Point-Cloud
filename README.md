# Transfer-Learning-Based-Semantic-Segmentation-for-3D-Object-detection-from-Point-Cloud
This code is based on the paper "Transfer Learning Based Semantic Segmentation for 3D Object detection from Point Cloud" (https://www.mdpi.com/1424-8220/21/12/3964)
# Collect custom Dataset
Collect 3D Lidar bag files using Ouster dataset (Or convert the KITTI Lidar dataset into bag files)
transform the pointcloud into BEV images
Label dataset for semantic segmentation using LableMe annotation tool
Convort labels into VOC format using script labelmetovoc.py

# Training
Train the model using python training.py

# Testing
To test the pretrained model on KITTI dataset run python Kitti_test.py
To test the pretrained model on KITTI dataset run python Ouster_birds_view.py
