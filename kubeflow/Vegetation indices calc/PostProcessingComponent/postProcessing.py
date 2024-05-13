import glob
import os
import argparse
from minioConnector import minioConector


if __name__ == "__main__":
    print("Starting Post Processing Component")
    print("Parsiranje Input Argumenata")
    parser = argparse.ArgumentParser(description='My program description')
    parser.add_argument('--bucket_name', type=str)
    parser.add_argument('--file_path', type=str)
    parser.add_argument('--minio_path', type=str)
    parser.add_argument('--new_location', type=str)
    args = parser.parse_args()
    print("Ended Parsing")
    bucket_name = args.bucket_name
    minio_path = args.minio_path
    file_path = args.file_path
    new_location = args.new_location

    print("Uploading to bucket: " + bucket_name)
    print("Uploading the following dir: " + file_path)
    print("Minio Path:" + minio_path)
    print(new_location)

    Minio_object= minioConector()
    Minio_object.uploadDirectory(file_path, bucket_name, minio_path)
    print("END OF POSTPROC")