## Dataset

We used the ScanNet Dataset. ScanNet is an RGB-D video dataset containing 2.5 million views in more than 1500 scans, annotated with 3D camera poses, surface reconstructions, and instance-level semantic segmentations. You can find the website of the Dataset [here](http://www.scan-net.org/#code-and-data) and the github [here](https://github.com/ScanNet/ScanNet).

Note that in order to dowload the dataset you need to ask for the authorization. Therefore we only put a sample of the dataset for both privicy and storage reason in the "/Dataset" folder.

## Additional Datasets

In order to use the UpRightNet model implementation from [zhengqili](https://github.com/zhengqili/UprightNet) we also had to use a pre-processed version of the ScanNet dataset. Download pre-processed InteriorNet and ScanNet, as well as their corresponding training/validation/testing txt files from [link](https://drive.google.com/drive/folders/1WdNAESqDYcUPQyXAW6PvlcdQIYlOEXIw). The dataset is very large therefore we only put a sample of it in the folder "/Dataset" for information purpose only.

## PoseLibUpgrade

In order to use the 3-pt RANSAC algorithm we upgrade the [PoseLib](https://github.com/vlarsson/PoseLib) library. 
Severals modification have been done, all the modify files are available in the folder "/PoseLibeUpgrade".
Note that in order to use our implementation of 3-pt RANSAC method you need to dowload the [PoseLib](https://github.com/vlarsson/PoseLib) library and change the according files before building (MAKE in c++) the library. We also modify the file that allow to use our implementation of PoseLib in Python.

### Gravity Estimation

The complete gravity estimation folder was originally structured from a clone of the github implementation of UpRightNet.

- The test.py file is a modified version of the test.py file in the original github to enable gravity vector prediction extraction and dumping the rot,roll,pitch error statistics in a pickle file.

- The image_labels_[0-9].pickle files are lists of numbers corresponding to the tags found by the yolov3 model in the images, each image_labels pickle file contains 500 elements in order of the test_scannet/test_scannet_normal_list_4.txt file, totaling 5000 images (full dataset).

- The results_final.pickle file contains a dictionnary of all the rot,roll,pitch errors in order of the test_scannet/test_scannet_normal_list_4.txt file

- The util folder contains:
    - a coco.names file a list of objects that the yolov3 model looks for in each image
    - a script.py file, an export of the ipynb notebook file used to generate all the results and statistics, this file performs the study
    - a yolov3.cfg file to load the yolov3 model with open-cv, a yolov3.weights file was also included but was too large to be upload, here is a polybox link to the used weights [yolov3.weights](https://polybox.ethz.ch/index.php/s/cOyTekyiPBC5jEd/download)
    
- The test_scannet folder contains:
    - a evaluable_scenes folder which is the dataset from the github implementation, we have selected scenes totaling over 5000 images to build our dataset
    - a generate_file_content.py file that generates the test_scannet_normal_list_4.txt file (or several files to split the dataset into several parts). It contains a list of paths which is then past to the data_loader-
    - a generate_prediction_errors.py file which iterates on all predicted gravity vectors and ground truth vector to compute an error and stores it in the approriate folder (predictions_for_mj which is used to run 3point and 5point with the predicted vectors)
    
### Relative Pose Estimation

As mentioned in the report, we used 3 differents methods during this project:

1. The 5-pt RANSAC Algorithm
2. The 3-pt RANSAC algorithm using ground truth gravity direction
3. The 3-pt RANSAC algorithm using estimated gravity direction

Therefor we provide you the 3 scripts (.py) used in order to compute the relative pose from the ScanetDataset:

1. 3p_poselib_upright.py: script computing the relative pose between using 3-pt RANSAC and ground truth gravity. The script run on a full scene, modifiy the scene path to get results for any scenes. You also need to generate the gravity estimation for the scene you want to test and link the path at the according place on the script.
2. 3p_poselib_upright_estimated.py: script computing the relative pose between using 3-pt RANSAC and estimated gravity from UpRightNetModel. The script run on a full scene, modifiy the scene path to get results for any scenes.
3. 5p_poselib.py: script computing the relative pose between using 5-pt RANSAC. The script run on a full scene, modifiy the scene path to get results for any scenes.

All files are available in the "/RelativePoseEstimation" folder.

### Benchmark & Analysis

The final part of the project was to analyse our results. To do so we used jupyter notebook rather than scripts as it is way more convenient for data and results investigation. We provide you 3 of the most important .ipynb we used to analyze our results:

1. benchmarking_and_analysis_3p_estimated.ipynb: you can find the analysis of the results for the 3-pt RANSAC algo using estimated gravity. We mostly get and aggregate results for all scenes, compute statistics on the R_err distribution and also compute the correlationa and provide a scatter plot between the gravity_error from gravity estimation results with UpRightNet and the R_err from relative pose estimation results using 3-pt RANSAC. 
2. benchmarking_and_analysis_3pVS5p.ipynb: you can find the analysis and comparison between the 5-pt RANSAC and the 3-pt RANSAC using ground truth. We plot severals distribution, analyze the running time, some correlation (R-err/running time, #key points/running time), the distribution or the R-err and more.
3. OPENCV_5p_Analysis.html: this file is our first attempt to apply the 5-pt RANSAC algo. We used it to try different type of descriptos (we finally choose SIFT) but also to determine the gap we will use between the frame when computing the relative pose. This file is a more personal one, used for investigation only at the beginning of the project to have a better understanding of the challenges and methods. This file is in .html format so it does not take much storage capicity as we are limited.

Note that for file 1 and 2 all the results are present and described in the final report. For the file 3, we decided to not talk about it in our final report as we used this personnaly to get a better understanding of the project and the 5-pt RANSAC method (with BF matching and SIFT descriptors). All files are availables in the "/Benchmark" folder. Note that you can also find some of the figures we used in our final report (generated by those ipynb)

### Results

We also provide you some partial example of the results you can get (only sample for storage capacity purpose). You can find:

1. scene0000_00_final_results_3p_gap32.txt: results for scene 0000_00 for relative pose estimation using 3-pt RANSAC and ground truth gravity (frame gap: 32)
2. scene0000_00_final_results_5p_gap32.txt : results for scene 0000_00 for relative pose estimation using 5-pt RANSAC (frame gap: 32)
3. scene0011_00_final_results_estimatedgravity_3p_gap32.txt: results for scene 0011_00 for relative pose estimation using 3-pt RANSAC and estimated gravity direction from UpRighNet results (frame gap: 32)
4. scene0011_00_predicted_gravity: folder where you can observe results obtained from UpRightNet. For an image id you have the estimated gravity direction and the error (compared to ground truth)
5. yolov3_output.png: this is the results table obtain from the Yolov3 analysis

Note that all the uploaded results are partial for storage purpose. You can find them in the "/Results" folder
