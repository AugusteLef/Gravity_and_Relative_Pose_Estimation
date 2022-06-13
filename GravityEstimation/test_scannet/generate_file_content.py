import os
import numpy as np


def generate_file_name(index):
    return '/content/gdrive/MyDrive/UprightNet/test_scannet/test_scannet_normal_list_final_'+str(index//500)+'.txt'

i = 0
path = "/content/gdrive/MyDrive/UprightNet/test_scannet/evaluable_scenes/"
for dirname in os.listdir(path):
        if os.path.isdir(path+dirname):
          for filename in os.listdir(path+"/"+dirname+"/normal_pair"):
            with open(generate_file_name(i), 'a') as f:
              f.write(path+dirname+"/normal_pair/"+filename+"\n")
              i+=1
              print(path+dirname+"/normal_pair/"+filename+"\n")
