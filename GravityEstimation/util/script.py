from google.colab import drive
drive.mount('/content/gdrive')

drive.mount('/content/gdrive', force_remount=True)

import sys
sys.path.append('/content/gdrive/My Drive/UprightNet')
sys.path.append('/content/gdrive/My Drive/')
import pandas as pd
import pickle
import cv2
from google.colab.patches import cv2_imshow
import numpy as np
import linecache
import re
from tabulate import tabulate

df2 = pd.read_pickle('/content/gdrive/MyDrive/UprightNet/results_final.pickle')

len(df2["pitch_e_list"])
!python /content/gdrive/MyDrive/UprightNet/test_scannet/generate_file_content.py #generates a txt file for the UpRightNet implemetation to properly load the daset

!python /content/gdrive/MyDrive/UprightNet/test.py --mode ResNet --dataset scannet #evaluates the model on our dataset

!python /content/gdrive/MyDrive/UprightNet/test_scannet/generate_prediction_errors.py #creates files with the prediction error for each gravity vector prediction

def get_tags_for_image(path):
  """_summary_

  Args:
      path (string): path of folder containing all images of the dataset

  Returns:
      list: returns a list as long as the number of files in the path folder and sub folders corresponding to objects found in the respective image
  """
  img_src = cv2.imread(path)
  classes_names = open('/content/gdrive/My Drive/util/coco.names').read().strip().split('\n') #list of tags taht the yolov3 model is looking for
  np.random.seed(42)
  colors_rnd = np.random.randint(0, 255, size=(len(classes_names), 3), dtype='uint8')

  net_yolo = cv2.dnn.readNetFromDarknet('/content/gdrive/My Drive/util/yolov3.cfg', '/content/gdrive/My Drive/util/yolov3.weights') #loads the pretrained yolov3 network architecture and weights
  net_yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
  net_yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

  ln = net_yolo.getLayerNames()

  l  = [item for sublist in net_yolo.getUnconnectedOutLayers() for item in sublist]
  ln = [ln[i - 1] for i in l]

  blob_img = cv2.dnn.blobFromImage(img_src, 1/255.0, (416, 416), swapRB=True, crop=False)


  net_yolo.setInput(blob_img)
  outputs = net_yolo.forward(ln)

  boxes = []
  confidences = []
  classIDs = []
  h, w = img_src.shape[:2]

  for output in outputs:
      for detection in output:
          scores_yolo = detection[5:]
          classID = np.argmax(scores_yolo)
          confidence = scores_yolo[classID]
          if confidence > 0.5:
              
              classIDs.append(classID)

  cv2.waitKey(0)
  cv2.destroyAllWindows()

  tags = set([classes_names[id] for id in classIDs])

  return tags


#dumps the list returned by the get_tag_for_images in a pickle file for future usage
with open('/content/gdrive/My Drive/UprightNet/test_scannet/test_scannet_normal_list_4.txt') as f:
    lines = np.array(f.readlines())
    splitted_list = np.split(lines, 10)

    for i in range(len(splitted_list)):
      if i>3:
          list_of_set_of_tags = [get_tags_for_image(p.replace("\n","")) for p in splitted_list[i]]
          with open('/content/gdrive/My Drive/UprightNet/image_labels_'+str(i)+'.pickle', 'wb') as f:
              pickle.dump(list_of_set_of_tags, f)
              print("dumped")
              print(str(i))
    
    print("ok")
  

eval_list_path = '/content/gdrive/MyDrive/UprightNet/test_scannet/test_scannet_normal_list_4.txt'

gravity_errors = list()

#writes the gravity prediction error in the corresponding file named [image_id]_error.txt
for index in range(5000):
  line = linecache.getline(eval_list_path, index+1)
  if line:
    sub_path = re.findall(r"scene[0-9]{0,10}_[0-9]{0,19}/normal_pair/[0-9]{0,10}.png",line)

    if sub_path:
          print(index)
          
          new_path = "/content/gdrive/MyDrive/UprightNet/test_scannet/predictions_for_mj/"+sub_path[0].replace("/normal_pair","").replace("png","txt")
          
          new_path_error = new_path.replace(".txt","_error.txt")

          with open(new_path_error) as f:
            error = float(f.readlines()[0])
            gravity_errors.append(error)
            

#dumps all the gravity error perdictions in a pickle file in order
with open('/content/gdrive/My Drive/UprightNet/gravity_errors.pickle', 'wb') as f:
              pickle.dump(gravity_errors, f)
              print("dumped")

print(len(gravity_errors))


labels = list()

#loads the image tags pickle file into  the labels variable
for j in range(10):
  file_path = "/content/gdrive/MyDrive/UprightNet/image_labels_"+str(j)+".pickle"
  data = pd.read_pickle(file_path)
  labels += data


print(labels)


print(len(labels))


list_of_set_of_tags = labels
len(list_of_set_of_tags)

#filters the image indices of the dataset per presence of a given object
indices = range(len(list_of_set_of_tags))
bed = [i for i in indices if "bed" in list_of_set_of_tags[i] ]
chair = [i for i in indices if "chair" in list_of_set_of_tags[i] ]
table = [i for i in indices if "diningtable" in list_of_set_of_tags[i] ]
sofa = [i for i in indices if "sofa" in list_of_set_of_tags[i] ]
none = [i for i in indices if "sofa" not in list_of_set_of_tags[i] and "bed" not in list_of_set_of_tags[i] and "chair" not in list_of_set_of_tags[i] and "diningtable" not in list_of_set_of_tags[i] ]


#holds rot,roll,pitch,gravity errors for images containing each type of object 
bed_rot_error = [df2["rot_e_list"][i] for i in bed]
chair_rot_error = [df2["rot_e_list"][i] for i in chair]
table_rot_error = [df2["rot_e_list"][i] for i in table]
sofa_rot_error = [df2["rot_e_list"][i] for i in sofa]
none_rot_error = [df2["rot_e_list"][i] for i in none]

bed_roll_error = [df2["roll_e_list"][i] for i in bed]
chair_roll_error = [df2["roll_e_list"][i] for i in chair]
table_roll_error = [df2["roll_e_list"][i] for i in table]
sofa_roll_error = [df2["roll_e_list"][i] for i in sofa]
none_roll_error = [df2["roll_e_list"][i] for i in none]

bed_pitch_error = [df2["pitch_e_list"][i] for i in bed]
chair_pitch_error = [df2["pitch_e_list"][i] for i in chair]
table_pitch_error = [df2["pitch_e_list"][i] for i in table]
sofa_pitch_error = [df2["pitch_e_list"][i] for i in sofa]
none_pitch_error = [df2["pitch_e_list"][i] for i in none]

bed_gravity_error = [gravity_errors[i] for i in bed]
chair_gravity_error = [gravity_errors[i] for i in chair]
table_gravity_error = [gravity_errors[i] for i in table]
sofa_gravity_error = [gravity_errors[i] for i in sofa]
none_gravity_error = [gravity_errors[i] for i in none]

#builds a table with mean, median, std for each of rot,roll,pitch,gravity error for all images of the dataset

main_table = list()
main_table.append(["Tag","mean_rot_e","median_rot_e","std_rot_e","mean_roll_e","median_roll_e","std_roll_e","mean_pitch_e","median_pitch_e","std_pitch_e","mean_gravity_e","median_gravity_e","std_gravity_e"]) #table header

def generate_line(tag,rot_values,roll_values,pitch_values, gravity_values):

    np_rot_values = np.array(rot_values)
    np_roll_values = np.array(roll_values)
    np_pitch_values = np.array(pitch_values)
    np_gravity_values = np.array(gravity_values)

    def generate_3_stats(arr):
      return [np.mean(arr),np.median(arr),np.std(arr)]

    return [tag]+generate_3_stats(np_rot_values)+generate_3_stats(np_roll_values)+generate_3_stats(np_pitch_values)+generate_3_stats(np_gravity_values) #generates a line of the table

main_table.append(generate_line("5k imgs",df2["rot_e_list"],df2["roll_e_list"],df2["pitch_e_list"],gravity_errors))



print(tabulate(main_table,headers ="firstrow",tablefmt="fancy_grid"))

#builds a table with mean, median, std for each of rot,roll,pitch,gravity error for all images of each class (presence of objects)

table = list()
table.append(["Tag","mean_rot_e","median_rot_e","std_rot_e","mean_roll_e","median_roll_e","std_roll_e","mean_pitch_e","median_pitch_e","std_pitch_e","mean_gravity_e","median_gravity_e","std_gravity_e"])

def generate_line(tag,rot_values,roll_values,pitch_values, gravity_values):

    np_rot_values = np.array(rot_values)
    np_roll_values = np.array(roll_values)
    np_pitch_values = np.array(pitch_values)
    np_gravity_values = np.array(gravity_values)

    def generate_3_stats(arr):
      return [np.mean(arr),np.median(arr),np.std(arr)]

    return [tag]+generate_3_stats(np_rot_values)+generate_3_stats(np_roll_values)+generate_3_stats(np_pitch_values)+generate_3_stats(np_gravity_values)

table.append(generate_line("bed",bed_rot_error,bed_roll_error,bed_pitch_error,bed_gravity_error))
table.append(generate_line("chair",chair_rot_error,chair_roll_error,chair_pitch_error, chair_gravity_error))
table.append(generate_line("table",table_rot_error,table_roll_error,table_pitch_error,table_gravity_error))
table.append(generate_line("sofa",sofa_rot_error,sofa_roll_error,sofa_pitch_error,sofa_gravity_error))
table.append(generate_line("none",none_rot_error,none_roll_error,none_pitch_error,none_gravity_error))




print(tabulate(table,headers ="firstrow",tablefmt="fancy_grid"))




