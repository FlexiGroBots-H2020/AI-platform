import json
import logging
import argparse
from pathlib import Path
import pandas as pd
import os
import urllib3
import certifi
from KubeFlow_Training_and_Testing import *

logging.basicConfig(level=logging.INFO)

def _train_model(args):

    logging.info('Loading download data')
    print("OVO JE ARGS.DATA")
    print(args.data)
    print("___________________777777____________________")
    rez = main( args.data, args.data, 0)
    print("????????????????????????????????????????????????")
    print(rez)

    with open(rez[0], 'w') as accuracy_file:
        accuracy_file.write(str(accuracy))
    #zovi dalje sa ovi
    logging.info('Saving model')


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
