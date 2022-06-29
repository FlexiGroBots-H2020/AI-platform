import logging
import kserve
from typing import Dict
import requests
import os
import tensorflow as tf
import certifi
import urllib3
from minio import Minio

logging.basicConfig(level=logging.INFO)


def getMinioClient( easier_user="minio-user", easier_password="minio-pass", 
                    easier_url="minio-cli.platform.flexigrobots-h2020.eu", minio_secure=True, minio_region='es'):

    minioClient = Minio(easier_url,
                    access_key=easier_user,
                    secret_key=easier_password,
                    secure=minio_secure,
                    region=minio_region)

    return minioClient

class KFServingModel(kserve.KFModel):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.ready = False
        # self.model_path = os.environ["MODEL_PATH"]+'/model.h5'

    def load(self):
        minioClient = getMinioClient()
        minioClient.fget_object('atos-demo-models', 'kubeflow_demo_model.h5', './kubeflow_demo_model.h5')
        self.model = tf.keras.models.load_model('./kubeflow_demo_model.h5')
        self.ready = True


    def predict(self, request: Dict) -> Dict:
        logging.info('KServe')
        logging.info(str(request))
        inputs = request['instances'][0]

        logging.info('Performing a prediction')
        prediction = self.model.predict(inputs)
        return {
                'prediction': prediction
                }

if __name__ == '__main__':
    
    logging.info("Starting KServe Demo")
    model = KFServingModel('demo-inference-service')
    model.load()
    kserve.KFServer(workers=1).start([model])