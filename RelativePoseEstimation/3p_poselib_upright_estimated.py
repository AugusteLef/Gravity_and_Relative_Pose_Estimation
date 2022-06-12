import numpy as np
import os
import sys
import cv2 as cv
from matplotlib import pyplot as plt
import math
from scipy.spatial.transform import Rotation as sci_R
import poselib
import time

#relative pose estimation
def estimate_relative_pose_from_correspondence(pts1, pts2, f_x, f_y, c_x, c_y):
    """ Estimate relative pose using the PoseLib upgraded implementation of 3-pt upright algorithm with RANSAC optimisation

    Args:
        pts1 (_type_): key points from image 1
        pts2 (_type_): key points from image 1
        f_x (_type_): camera intrinsic
        f_y (_type_): camera intrinsic
        c_x (_type_): camera intrinsic
        c_y (_type_): camera intrinsic

    Returns:
        _type_: relative pose and inliers
    """
    
    width = 1296
    height = 968
    camera = {'model': 'PINHOLE', 'width': width, 'height': height, 'params': [f_x, f_y, c_x, c_y]}
    output = poselib.estimate_relative_pose_3ptupright(pts1, pts2, camera, camera, {}, {})
    return output


def pose_error(R1, R2, t1, t2):
    """Method to compute the error in degrees between ground truth and estimated relative pose

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

def get_relative_pose(path_img1, path_img2, path_pose1, path_pose2, g_path1, g_path2, path_K):
    """Method that prepare the images and the data in order to compute the relative poses between both by callin the according method.
    Task: get keypoints, bruteforce matching on key points, arrange key points location according to gravity direction, call the estimate relative pose method.

    Args:
        path_img1 (_type_): path to the first image
        path_img2 (_type_): path to the second image
        path_pose1 (_type_): path to the pose of the first image to get gravity
        path_pose2 (_type_): path to the pose of the first image to get gravity
        g_path1 (_type_): path to the estimated gravity for image 1 
        g_path2 (_type_): path to the estimated gravity for image 2
         
        path_K (_type_): path to the intrinsic (same for both images as we use images taken with the same camera)

    Returns:
        _type_: The error between the gt and est. relative pose of the two images given as inputs
    """
    
    # load images
    img1 = cv.imread(path_img1,0)  #queryimage # left image
    img2 = cv.imread(path_img2,0) #trainimage # right 
    
    # find the keypoints and descriptors with SIFT
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
 
    # read the intrinsic matrix
    K = np.loadtxt(intrinsic_path, dtype=float)
    K = np.delete(K,3,0)
    K = np.delete(K,3,1) 
    
    # brute force matching with Knn (k=2)
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
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
    

    ##### HERE WE WANT TO APPLY GRAVITY TO OUR KEY POINTS IN ORDER TO GET UPRIGHT IMAGES
    # first add a coordinate to have homogenous coordinates
    keyp1_hom = np.column_stack((pts1,[1]*len(pts1)))
    keyp2_hom = np.column_stack((pts2,[1]*len(pts2)))

    # convert homogenous keypoints to ray vectors
    keyp1_cam = [np.matmul(K,keyp1_hom) for keyp1_hom in keyp1_hom]
    keyp2_cam = [np.matmul(K,keyp2_hom) for keyp2_hom in keyp2_hom] 
    
    #get gravity vector for picture (estimated gravity version)
    with open(g_path1, 'r') as f:
        g_row = f.readlines()
        gprime, gsecond, gthird = g_row[0].split()
        g1 = np.array((np.array(gprime), np.array(gsecond), np.array(gthird)), dtype=float)
    print(g1)

    # apply gravity rotation to each keypoints
    rot_axis1 = np.array([-g1[2]/np.linalg.norm(g1),0,g1[0]/np.linalg.norm(g1)],dtype=object)
    rot_angle1 = np.arccos(g1[1]/np.linalg.norm(g1))
    rot_matrix1 = np.asarray(sci_R.from_rotvec([rot_axis1 * rot_angle1],degrees=True).as_matrix())
    keyp1_cam = [np.matmul(rot_matrix1,keyp1_cam) for keyp1_cam in keyp1_cam]
    print(keyp1_cam)
    keyp1_hom = [np.matmul(np.linalg.inv(K),keyp1_cam) for keyp1_cam in keyp1_cam[0]]

    pts1 = np.asarray([x[0:2]/x[2] for x in keyp1_hom], dtype=np.float64)
    #pts1_hom = np.column_stack((pts1,[1]*len(pts1)))

    #get gravity vector for picture 2(estimated gravity version)
    with open(g_path2, 'r') as f:
        g_row = f.readlines()
        gprime, gsecond, gthird = g_row[0].split()
        g2 = np.array((np.array(gprime), np.array(gsecond), np.array(gthird)), dtype=float)
    
    rot_axis2 = np.array([-g2[2]/np.linalg.norm(g2),0,g2[0]/np.linalg.norm(g2)],dtype=object)
    rot_angle2 = np.arccos(g2[1]/np.linalg.norm(g2))
    rot_matrix2 = np.asarray(sci_R.from_rotvec([rot_axis2 * rot_angle2],degrees=True).as_matrix())
    keyp2_cam = [np.matmul(rot_matrix2,keyp2_cam) for keyp2_cam in keyp2_cam]

    keyp2_hom = [np.matmul(np.linalg.inv(K),keyp2_cam) for keyp2_cam in keyp2_cam[0]]

    pts2 = np.asarray([x[0:2]/x[2] for x in keyp2_hom], dtype=np.float64)

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
    R = rot_matrix1[0] @ R
    R = R @ rot_matrix2[0].T 

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
# add try/catch to avoid error
# modfiy directory path, everything else should be done automatically


# GLOBAL VARIABLE (CHANGE ACCORDING TO YOUR NEEDS)
directory = r'/home/steineja/scratch_slurm/data/ScanNet/scans/'
######
estimated_directory = r'/home/steineja/scratch_slurm/data/ScanNet/predictions_for_mj/'
######
gap = 32
scene = "scene0011_00" #choose wisely!!!

# Don't modify from there
actual_folder_directory = ""
results = []
output_name = "final_results_estimatedgravity_3p_gap" + str(gap) + ".txt"

# FOR LOOP TO ITERATE OVER IMAGES AND COMPUTE THE RELATIVE POSE ESTIMATION
for scan in os.listdir(directory):
    if scan.startswith(scene):
        actual_folder_directory = os.path.join(directory, scan)
        
        #####
        estimated_folder = os.path.join(estimated_directory, scan)
        if not os.path.isdir(estimated_folder):
           print("WARNING: NO AVAILABLE ESTIMATED GRAVITY FOR SCENE :" + str(scan))
        #####
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
                ######
                base_gravity_path = os.path.join(estimated_folder, img[0:-4]+'.txt')
                ######
                number = base_number + gap
                str_number = str(number)
                img_path = os.path.join(color_folder,str_number+'.jpg')
                pose_path = os.path.join(pose_folder, str_number+'.txt')
                ######
                gravity_path = os.path.join(estimated_folder, str_number+'.txt')
                ######
                print('base gravity path:', base_gravity_path)
                print('gravity path:', gravity_path)

                if os.path.isfile(img_path) and os.path.isfile(gravity_path):
                    start = time.time()
                    R,t, k1, k2 = get_relative_pose(base_img_path, img_path, base_pose_path, pose_path, base_gravity_path, gravity_path, intrinsic_path)
                    end = time.time()
                    total = end-start
                    result = [base_number, number, R, t, k1, k2, total]
                    results.append(result)
                    print('get relative pose')

                    
        output_path = estimated_pose_path + '/' + scene + "_" +output_name
        np.savetxt(output_path, results, delimiter=' ')