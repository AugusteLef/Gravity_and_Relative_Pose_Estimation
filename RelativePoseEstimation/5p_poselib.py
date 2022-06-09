import numpy as np
import os
import sys
import cv2 as cv
from matplotlib import pyplot as plt
import math
import pandas as pd
from scipy.spatial.transform import Rotation as sci_R
import poselib
import time

def estimate_relative_pose_from_correspondence(pts1, pts2, f_x, f_y, c_x, c_y):
    """Estimate relative pose using the PoseLib upgraded implementation of 5-pt algorithm with RANSAC optimisation

    Args:
        pts1 (_type_): keypoints from image 1
        pts2 (_type_): keypoints from image 2
        f_x (_type_): camera intrinsics
        f_y (_type_): camera intrinsics
        c_x (_type_): camera intrinsics
        c_y (_type_): camera intrinsics

    Returns:
        _type_: pose and inliners
    """
    
    #height and width from properties jpg files (1296x986)
    width = 1296
    height = 968
    
    camera = {'model': 'PINHOLE', 'width': width, 'height': height, 'params': [f_x, f_y, c_x, c_y]}

    #relative pose estimation
    output = poselib.estimate_relative_pose(pts1, pts2, camera, camera, {'max_epipolar_error': 1.0}, {})

    return output


def pose_error(R1, R2, t1, t2):
    """_summary_

    Args:
        R1 (_type_): Est. Rotation matrix
        R2 (_type_): GT Rotation matrix
        t1 (_type_): Est. Translation vector
        t2 (_type_): GT Translation vector

    Returns:
        _type_: Rotation and Translation errors
    """
    eps = 1e-15
    
    t = t1.flatten()
    t_gt = t2.flatten()
    
    R2R1 = np.dot(R1, np.transpose(R2))
    cos_angle = max(min(1.0, 0.5 * (np.trace(R2R1) - 1.0)), -1.0)
    err_r = math.acos(cos_angle)
    
    t = t / (np.linalg.norm(t) + eps)
    t_gt = t_gt / (np.linalg.norm(t_gt) + eps)
    loss_t = np.maximum(eps, (1.0 - np.sum(t * t_gt)**2))
    err_t = np.arccos(np.sqrt(1 - loss_t))
    
    if np.sum(np.isnan(err_r)) or np.sum(np.isnan(err_t)):
        print(R1, t_gt, R2, t)
        import IPython
        IPython.embed()
        
    return err_r * 180.0 / math.pi, err_t * 180.0 / math.pi

def get_relative_pose(path_img1, path_img2, path_pose1, path_pose2, path_K):
    """Method that prepare the images and the data in order to compute the relative poses between both by callin the according method.
    Task: get keypoints, bruteforce matching on key points, call the estimate relative pose method.

    Args:
        path_img1 (_type_): path to the first image
        path_img2 (_type_): path to the second image
        path_pose1 (_type_): path to the pose 
        path_pose2 (_type_): path to the pose 
        path_K (_type_): path to the intrinsic (same for both images as we use images taken with the same camera)

    Returns:
        _type_: The error between the gt and est. relative pose of the two images given as inputs
    """
    # read the images
    img1 = cv.imread(path_img1,0)  #queryimage # left image
    img2 = cv.imread(path_img2,0) #trainimage # right 
    
    # find the keypoints and descriptors with SIFT
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    
    #Brute force matching with Knn (k=2)
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper$
    ratio = 0.8
    for i,(m,n) in enumerate(matches):
        if m.distance < ratio*n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    if len(pts1) < 3 or len(pts2) < 3:
        print('not enough keypoints')
        return [0,0,0,0]

    nbr_keypoints1 = len(pts1)
    nbr_keypoints2 = len(pts2)
    
    # transform points to np format
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    

    

    #Get intrisinc matrix
    #preparing everything for poselib.estimate_relative_pose
    K = np.loadtxt(intrinsic_path, dtype=float)
    K = np.delete(K,3,0)
    K = np.delete(K,3,1)

    #intrinsics
    Camera_intrinsics = np.loadtxt(intrinsic_path, dtype=float)
    f_x = Camera_intrinsics[0, 0]
    f_y = Camera_intrinsics[1, 1]
    c_x = Camera_intrinsics[0, 2]
    c_y = Camera_intrinsics[1, 2]
        
    # get estimated relative pose
    pose, inliers = estimate_relative_pose_from_correspondence(pts1,pts2, f_x, f_y, c_x, c_y)
    if pose == []:
        print('failed pose estimation')
        return [0,0,0,0] 

    R = pose.R
    t = pose.t
    

    #relative pose GT from absolute pose GT
    R1 = np.loadtxt(path_pose1, usecols=range(3), max_rows=3)
    R2 = np.loadtxt(path_pose2, usecols=range(3), max_rows=3)
    t1 = np.loadtxt(path_pose1, usecols=3, max_rows=3)
    t2 = np.loadtxt(path_pose2, usecols=3, max_rows=3)
    R_gt = np.matmul(R2, R1.transpose())
    t_gt = np.matmul(t2 - R2, np.matmul(R1.transpose(), t1))
    t_gt = t_gt / np.linalg.norm(t_gt)
    if R_gt == 'inf' or t_gt == 'nan':
        print('invalid ground truth')
        return [0,0,0,0]


    R_err, t_err = pose_error(R, R_gt, t, t_gt)
    print('\nPose Error in degrees for R and t:')
    print(R_err, t_err)
    return R_err, t_err, nbr_keypoints1, nbr_keypoints2


# GLOBAL VARIABLE (CHANGE ACCORDING TO YOUR NEEDS)
directory = r'/home/steineja/scratch_slurm/data/ScanNet/scans/'
gap = 32
scene = "scene0000_00"

# Don't modify from there
actual_folder_directory = ""
results = []
output_name = "final_results_3p_gap" + str(gap) + ".txt"

# FOR LOOP TO ITERATE OVER IMAGES AND COMPUTE THE RELATIVE POSE ESTIMATION
for scan in os.listdir(directory):
    if scan.startswith(scene):
        actual_folder_directory = os.path.join(directory, scan)
        print("We are working in: " + actual_folder_directory)
        intrinsic_path = actual_folder_directory + '/intrinsic/intrinsic_color.txt'
        color_folder = actual_folder_directory + '/color'
        pose_folder = actual_folder_directory + '/pose'
        estimated_pose_path = pose_folder + '/estimated_pose'
        if not os.path.isdir(estimated_pose_path):
            os.mkdir(estimated_pose_path)
        counter = 0
        for img in os.listdir(color_folder): #["0.jpg", "1.jpg", "2.jpg", "3.jpg"]: #in os.listdir(color_folder):
            counter += 1
            print('\n',counter,'\n')
            if img.endswith(".jpg"):
                base_number = int(img[0:-4])
                base_img_path = os.path.join(color_folder, img)
                base_pose_path = os.path.join(pose_folder, img[0:-4]+'.txt')
                number = base_number + gap
                str_number = str(number)
                img_path = os.path.join(color_folder,str_number+'.jpg')
                pose_path = os.path.join(pose_folder, str_number+'.txt')
                if os.path.isfile(img_path):
                    start = time.time()
                    R,t, k1, k2 = get_relative_pose(base_img_path, img_path, base_pose_path, pose_path, intrinsic_path)
                    end = time.time()
                    total = end-start
                    result = [base_number, number, R, t, k1, k2, total]
                    results.append(result)
                    
        output_path = estimated_pose_path + '/' + scene + "_" + output_name
        np.savetxt(output_path, results, delimiter=' ')