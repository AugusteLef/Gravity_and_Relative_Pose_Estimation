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
3. OPENCV_5p_Analysis.html: this file is our first attempt to apply the 5-pt RANSAC algo. We used it to try different type of descriptos (we finally choose SIFT) but also to determine the gap we will use between the frame when computing the relative pose. This file is a more personal one, used for investigation only at the beginning of the project to have a better understanding of the challenges and methods.

Note that for file 1 and 2 all the results are present and described in the final report. For the file 3, we decided to not talk about it in our final report as we used this personnaly to get a better understanding of the project and the 5-pt RANSAC method (with BF matching and SIFT descriptors). All files are availables in the "/Benchmark" folder. Note that you can also find some of the figures we used in our final report (generated by those ipynb)
