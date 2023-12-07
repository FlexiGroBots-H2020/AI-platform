import sys
import pandas as pd
from os import listdir
from os import getcwd
from os.path import isfile, join
import argparse
from pathlib import Path
from minioConector import minioConector
import datetime


#pipeline purpose only



parser = argparse.ArgumentParser(description='My program description')
parser.add_argument('--bucket_name', type=str)
parser.add_argument('--log_location_tiff', type=str)
parser.add_argument('--log_location_colored_tiff', type=str)
parser.add_argument('--log_location_shp', type=str)
parser.add_argument('--new_location', type=str)

args = parser.parse_args()
bucket_name =  args.bucket_name
log_location1 = args.log_location_tiff
log_location2 = args.log_location_colored_tiff
log_location3 = args.log_location_shp
path = args.new_location
# path = '/home/tloken/biosens/borovnice/DataTest/final_results_folder/test_parcel.tif'
# bucket_name = 'test-result'
upload_tmp = r"/mnt/model/blueberry-row-detection-model.pt"
print(path)
print("current path: ", getcwd())
from os import chdir

#chdir(path)
#sys.path.append(path)
print(sys.path)
#print("current path: ", getcwd())
result_path = None
list_dir_path = listdir(path)
print(list_dir_path)
for d in list_dir_path:
    if d[:3] == 'rgb': 
        result_path = path+d
        
time = datetime.datetime.now()
time = str(time).replace(' ',';')

Minio_object= minioConector()
#Minio_object.uploadFiles(bucket_name, 'unetpp_test_parcel' + time + '.tif', log_location1)
#Minio_object.uploadFiles(bucket_name, 'rgb_masked_with_red' + time + '.tif', log_location2)
#Minio_object.uploadFiles(bucket_name, 'unetpp_test_parcel' + time + '.shp', log_location3)
#Minio_object.uploadFiles(bucket_name, 'unetpp_test_parcel' + time + '.dbf', log_location3[:-3]+"dbf")
#Minio_object.uploadFiles(bucket_name, 'unetpp_test_parcel' + time + '.shx', log_location3[:-3]+"shx")
print("bucket namet: ", bucket_name)
print("result patch: ",result_path)
Minio_object.uploadDirectory("/mnt/DataTest/final_results_folder",bucket_name,"final_results_folder")
#/blueberry-results//mnt/DataTest/final_results_folder//DataTest/kanali_npy/test_ch_red_edge_croped.npy    
# Minio_object.uploadFiles(bucket_name,"blueberry_row_detection_model.tif","rgb_masked_with_red_belanovica.tif")
#Minio_object.uploadFiles(bucket_name, "final_results_folder/unetpp_test_parcel.tiff", "/mnt/DataTest/final_results_folder/unetpp_test_parcel_belanovica.tiff",my_content_type="image/tiff")
#Minio_object.uploadFiles(bucket_name, 'rgb_masked_with_red' + '.tif', log_location2)
# Minio_object.uploadFiles(bucket_name, 'probafoldera/blueberry_row_detection_model' + '.pt', "/mnt/DataTest/final_results_folder/blueberry_row_detection_model.pt")
# Minio_object.uploadFiles(bucket_name, 'probafoldera/unetpp_test_parcel' + '.shp', log_location3)
# Minio_object.uploadFiles(bucket_name, 'probafoldera/unetpp_test_parcel' + '.dbf', log_location3[:-3]+"dbf")
# Minio_object.uploadFiles(bucket_name, 'probafoldera/unetpp_test_parcel' + '.shx', log_location3[:-3]+"shx")





