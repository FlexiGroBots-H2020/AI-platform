from minioConector import minioConector
import os
import argparse
from pathlib import Path
from os import listdir
from os.path import isfile, join


parser = argparse.ArgumentParser(description='parser')
parser.add_argument('--bucket_name', type=str)
parser.add_argument('--save_location', type=str)
parser.add_argument('--output1_path', type=str)
args = parser.parse_args()
bucket_name = args.bucket_name
save_location = args.save_location

Minio_object = minioConector()

print("========================")
print("all buckets")
print(Minio_object.listBuckets())
print("========================")


#TODO: download odakle da vuce podatke
#TODO: download gde da cuva na HDD

if os.path.isdir(save_location + '/Data'):
    print("========================Data already exists========================")
else:
    print("========================Data doesn't exists========================")
    print("========================Start downloading data========================")
    print("...")
    Minio_object.downloadBucket(bucket_name,save_location)
    print("========================Finish downloading data========================")



# Minimalan kod da bi postojao pipeline, ne diraj
#empty_cache moze kasnije da sluzi da prosledjuje lokacije
def do_work(save_location):
    _ = output1_path.write(save_location + '/Data')## TODO: napisati na kojoj lokaciji je sacuvao fajlove



Path(args.output1_path).parent.mkdir(parents=True, exist_ok=True)
with open(args.output1_path, 'w') as output1_path:
    do_work(save_location)
