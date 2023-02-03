import sys
import pandas as pd
from os import listdir
from os.path import isfile, join
import argparse
from pathlib import Path
from minioConector import minioConector
import datetime


#pipeline purpose only



parser = argparse.ArgumentParser(description='My program description')
parser.add_argument('--new_location', type=str)
parser.add_argument('--bucket_name', type=str)
parser.add_argument('--log_location_tiff', type=str)
parser.add_argument('--log_location_colored_tiff', type=str)
parser.add_argument('--log_location_shp', type=str)

args = parser.parse_args()
path = args.new_location
bucket_name = args.bucket_name
log_location1 = args.log_location_tiff
log_location2 = args.log_location_colored_tiff
log_location3 = args.log_location_shp
# path = '/home/tloken/biosens/borovnice/DataTest/final_results_folder/test_parcel.tif'
# bucket_name = 'test-result'

sys.path.append(path)

time = datetime.datetime.now()
time = str(time).replace(' ',';')

Minio_object= minioConector()
Minio_object.uploadFiles(bucket_name, 'unetpp_test_parcel' + time + '.tif', log_location1)
Minio_object.uploadFiles(bucket_name, 'rgb_masked_with_red' + time + '.tif', log_location2)
Minio_object.uploadFiles(bucket_name, 'unetpp_test_parcel' + time + '.shp', log_location3)
Minio_object.uploadFiles(bucket_name, 'unetpp_test_parcel' + time + '.dbf', log_location3[:-3]+"dbf")
Minio_object.uploadFiles(bucket_name, 'unetpp_test_parcel' + time + '.shx', log_location3[:-3]+"shx")





