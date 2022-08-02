import os
from dotenv import load_dotenv
import json
import argparse
import logging
import urllib3
import certifi
from pathlib import Path
import pandas as pd
from minio import Minio


logging.basicConfig(level=logging.INFO)

def getMinioClient( easier_user, easier_password, easier_url="minio-cli.platform.flexigrobots-h2020.eu", minio_secure=True, minio_region='es'):
    
    minioClient = Minio(easier_url, access_key=easier_user, secret_key=easier_password, secure=True, region='es')
    return minioClient

def _download_data(args):
    logging.info("Downloading Data")
    load_dotenv("../.env")
    minioClient = getMinioClient(easier_user=os.getenv('MINIO_ACCESS_KEY'),easier_password=os.getenv('MINIO_SECRET_KEY'))
    obj = minioClient.get_object('atos-demo-data', 'synthetic_data_with_target.csv')
    data = pd.read_csv(obj, index_col=0)
    data.to_csv(args.data)
    logging.info('Data Downloaded')

if __name__ == '__main__':
    
    # This component does not receive any input
    # it only outpus one artifact which is `data`.
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str)
    
    args = parser.parse_args()
    # Creating the directory where the output file will be created 
    # (the directory may or may not exist).
    Path(args.data).parent.mkdir(parents=True, exist_ok=True)

    _download_data(args)
    