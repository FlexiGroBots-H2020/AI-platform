import json
import logging
import argparse
from pathlib import Path
import pandas as pd
import os
import urllib3
import certifi
import minio
import tensorflow as tf
from minio import Minio
from tensorboard.plugins.hparams import api as hp



def getMinioClient( easier_user="minio-user", easier_password="minio-pass",
                    easier_url="minio-cli.platform.flexigrobots-h2020.eu", minio_secure=True, minio_region='es'):

    minioClient = Minio(easier_url,
                    access_key=easier_user,
                    secret_key=easier_password,
                    secure=True, region='es')
    return minioClient


logging.basicConfig(level=logging.INFO)

def _train_model(args):

    logging.info('Loading preprocessed data')
    with open(args.data) as data_file:
        data = json.load(data_file)
    data = json.loads(data)
    x_train = pd.read_json(data['x_train'])
    y_train = pd.read_json(data['y_train'])
    x_test = pd.read_json(data['x_test'])
    y_test = pd.read_json(data['y_test'])
    
    n_classes = y_train['target'].unique().shape[0]
    logging.info('Data laoded')
    hparams = {
        "num_units": 16,
        "dropout": 0.1,
        "optimizer": "adam",
        "epochs": 10
    }
    
    logging.info('Neural network definition')
    
    model = tf.keras.models.Sequential([
                        tf.keras.layers.Dense(hparams['num_units'],
                                            input_dim=x_train.shape[1],
                                            activation=tf.nn.relu),
                        tf.keras.layers.Dropout(hparams['dropout']),
                        tf.keras.layers.Dense(n_classes, activation=tf.nn.softmax),
                        ])
    model.compile(
        optimizer=hparams["optimizer"],
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
                                                    log_dir='/'
                                                    )
    logging.info('Start model training')
    model.fit(x_train, y_train,
                epochs=hparams["epochs"],
                callbacks=[
                        tensorboard_callback,
                        hp.KerasCallback('/', hparams)]) # Run with 1 epoch to speed things up for demo purposes
    logging.info('Perform model evaluation')
    _, accuracy = model.evaluate(x_test, y_test)
    
    
    with open(args.accuracy, 'w') as accuracy_file:
        accuracy_file.write(str(accuracy))

    # minioClient.
    logging.info('Saving model')
    model.save(args.model)
    minioClient = getMinioClient()
    minioClient.fput_object("atos-demo-models", "kubeflow_demo_model.h5", args.model)
    logging.info('Model saved')

if __name__ == '__main__':
    # Defining and parsing the command-line arguments
    parser = argparse.ArgumentParser(description='My program description')
    parser.add_argument('--data', type=str)
    parser.add_argument('--model', type=str, default='model.h5')
    parser.add_argument('--accuracy', type=str)
    args = parser.parse_args()

    # Creating the directory where the output file will be created (the directory may or may not exist).
    Path(args.accuracy).parent.mkdir(parents=True, exist_ok=True)
    
    _train_model(args)