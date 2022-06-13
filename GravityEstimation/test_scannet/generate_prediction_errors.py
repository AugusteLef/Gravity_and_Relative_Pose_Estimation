import linecache
import os
import numpy as np
import re

eval_list_path = '/content/gdrive/MyDrive/UprightNet/test_scannet/test_scannet_normal_list_4.txt'



for index in range(5000):
  line = linecache.getline(eval_list_path, index+1)
  if line:
    sub_path = re.findall(r"scene[0-9]{0,10}_[0-9]{0,19}/normal_pair/[0-9]{0,10}.png",line)

    if sub_path:
          print(index)
          new_path = "/content/gdrive/MyDrive/UprightNet/test_scannet/predictions_for_mj/"+sub_path[0].replace("/normal_pair","").replace("png","txt")
          gt_vector_path = line.replace("/normal_pair","/gt_poses").replace(".png\n",".txt")
          vector_text = linecache.getline(gt_vector_path, 3)
          c1,c2,c3 = vector_text.split(" ")
          gt_g_vector = [float(c1),float(c2),float(c3)]
          p1,p2,p3 = linecache.getline(new_path, 1).split(" ")
          pred_g_vector = [float(p1),float(p2),float(p3)]
          error = np.linalg.norm(np.array(pred_g_vector)-np.array(gt_g_vector))
          new_path_error = new_path.replace(".txt","_error.txt")
          with open(new_path_error, 'w') as temp_file:
                temp_file.write(str(error))
    #if not os.path.exists(re.findall(r".*scene[0-9]{0,10}_[0-9]{0,19}",new_path)[0]):
                #os.mkdir(re.findall(r".*scene[0-9]{0,10}_[0-9]{0,19}",new_path)[0])

    #with open(new_path, 'w') as temp_file:
                #temp_file.write(str(est_up_n_list[index][0])+" "+str(est_up_n_list[index][1])+" "+str(est_up_n_list[index][2]))