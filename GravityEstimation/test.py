from __future__ import division
import time
import torch
import numpy as np
from torch.autograd import Variable
import models.networks
from options.test_options import TestOptions 
import sys
from data.data_loader import *
from models.models import create_model
import random
import os
#from tensorboardX import SummaryWriter
import pickle
import linecache
import re

print("running")
with open('test.txt', 'w') as f:
    f.write("test")

EVAL_BATCH_SIZE = 1
opt = TestOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

root = '/'

eval_list_path = ""
test_dataset = ""
if opt.dataset == 'interiornet':
    eval_list_path = root + '/phoenix/S6/wx97/interiornet/test_interiornet_normal_list.txt'
    eval_num_threads = 3
    test_data_loader = CreateInteriorNetryDataLoader(opt, eval_list_path, 
                                                    False, EVAL_BATCH_SIZE, 
                                                    eval_num_threads)
    test_dataset = test_data_loader.load_data()
    test_data_size = len(test_data_loader)
    print('========================= InteriorNet Test #images = %d ========='%test_data_size)


elif opt.dataset == 'scannet':
    eval_list_path = root + '/content/gdrive/MyDrive/UprightNet/test_scannet/test_scannet_normal_list_4.txt'
    eval_num_threads = 2
    test_data_loader = CreateScanNetDataLoader(opt, eval_list_path, 
                                                    False, EVAL_BATCH_SIZE, 
                                                    eval_num_threads)
    test_dataset = test_data_loader.load_data()
    test_data_size = len(test_data_loader)
    print('========================= ScanNet eval #images = %d ========='%test_data_size)


else:
    print('INPUT DATASET DOES NOT EXIST!!!')
    sys.exit()

model = create_model(opt, _isTrain=False)
model.switch_to_train()

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
global_step = 0

print("aaa")
print(len(test_dataset))

def test_numerical(model, dataset, global_step):
    rot_e_list = []
    roll_e_list = []
    pitch_e_list = []
    est_up_n_list = []

    count = 0.0

    model.switch_to_eval()

    count = 0

    for i, data in enumerate(dataset):
        print(i)
        stacked_img = data[0]
        targets = data[1]

        rotation_error, roll_error, pitch_error, est_up_n = model.test_angle_error(stacked_img, targets) #modified the test_angle_error method to return the estimated gravity vector

        est_up_n_list.append(est_up_n.tolist())
        rot_e_list = rot_e_list + rotation_error    
        roll_e_list = roll_e_list + roll_error
        pitch_e_list = pitch_e_list + pitch_error

        rot_e_arr = np.array(rot_e_list)
        roll_e_arr = np.array(roll_e_list)
        pitch_e_arr = np.array(pitch_e_list)

        mean_rot_e = np.mean(rot_e_arr)
        # median_rot_e = np.median(rot_e_arr)
        std_rot_e = np.std(rot_e_arr)

        mean_roll_e = np.mean(roll_e_arr)
        # median_roll_e = np.median(roll_e_arr)
        std_roll_e = np.std(roll_e_arr)

        mean_pitch_e = np.mean(pitch_e_arr)
        # median_pitch_e = np.median(pitch_e_arr)
        std_pitch_e = np.std(pitch_e_arr)


    rot_e_arr = np.array(rot_e_list)
    roll_e_arr = np.array(roll_e_list)
    pitch_e_arr = np.array(pitch_e_list)

    mean_rot_e = np.mean(rot_e_arr)
    median_rot_e = np.median(rot_e_arr)
    std_rot_e = np.std(rot_e_arr)

    mean_roll_e = np.mean(roll_e_arr)
    median_roll_e = np.median(roll_e_arr)
    std_roll_e = np.std(roll_e_arr)

    mean_pitch_e = np.mean(pitch_e_arr)
    median_pitch_e = np.median(pitch_e_arr)
    std_pitch_e = np.std(pitch_e_arr)


    results = {"rot_e_list":rot_e_list,"roll_e_list":roll_e_list,"pitch_e_list":pitch_e_list,
    "stat_rot_e_list":(mean_rot_e,median_rot_e,std_rot_e),
    "stat_roll_e_list":(mean_roll_e,median_roll_e,std_roll_e),
    "stat_std_pitch_e_list":(mean_pitch_e,median_pitch_e,std_pitch_e)}
    #writes the rot,roll,pitch errors in a pickle file
    with open('/content/gdrive/My Drive/UprightNet/results_final.pickle', 'wb') as f:
        pickle.dump(results, f)
        print("dumped")
        print("ok")
    print(est_up_n_list)
    #dumps the estimated vector in the corresponding file at the corresponding subfolder, paths are built with regex processing from the original dataset paths file
    for index in range(len(est_up_n_list)):
        line = linecache.getline(eval_list_path, index+1)
        sub_path = re.findall(r"scene[0-9]{0,10}_[0-9]{0,19}/normal_pair/[0-9]{0,10}.png",line)
        if sub_path:
          new_path = "/content/gdrive/MyDrive/UprightNet/test_scannet/predictions_for_mj/"+sub_path[0].replace("/normal_pair","").replace("png","txt")
          if not os.path.exists(re.findall(r".*scene[0-9]{0,10}_[0-9]{0,19}",new_path)[0]):
            os.mkdir(re.findall(r".*scene[0-9]{0,10}_[0-9]{0,19}",new_path)[0])
          with open(new_path, 'w') as temp_file:
            temp_file.write(str(est_up_n_list[index][0])+" "+str(est_up_n_list[index][1])+" "+str(est_up_n_list[index][2]))
            #pass

    print('======================= FINAL STATISCIS ==========================')
    print('mean_rot_e ', mean_rot_e)
    print('median_rot_e ', median_rot_e)
    print('std_rot_e ', std_rot_e)

    print('mean_roll_e ', mean_roll_e)
    print('median_roll_e ', median_roll_e)
    print('std_roll_e ', std_roll_e)

    print('mean_pitch_e ', mean_pitch_e)
    print('median_pitch_e ', median_pitch_e)    
    print('std_pitch_e ', std_pitch_e)


test_numerical(model, test_dataset, global_step)
