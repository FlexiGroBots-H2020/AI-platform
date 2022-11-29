from minioConector import minioConector
import os
import argparse
from pathlib import Path

Minio_object = minioConector()

print("========================")
print("all buckets")
print(Minio_object.listBuckets())
print("========================")


if os.path.isdir('/mnt/Data'):
    print("========================Data already exists========================")
else:
    print("========================Data doesn't exists========================")
    print("========================Start downloading data========================")
    print("...")
    Minio_object.downloadBucket("blueberry-data","/mnt")
    print("========================Finish downloading data========================")



# Minimalan kod da bi postojao pipeline, ne diraj
#empty_cache moze kasnije da sluzi da prosledjuje lokacije
def do_work(output1_file):
    _ = output1_file.write("empty_cache")

parser = argparse.ArgumentParser(description='My program description')

parser.add_argument('--output1-path', type=str, help='Path of the local file where the Output 1 data should be written.')
args = parser.parse_args()

Path(args.output1_path).parent.mkdir(parents=True, exist_ok=True)
with open(args.output1_path, 'w') as output1_file:
    do_work(output1_file)
