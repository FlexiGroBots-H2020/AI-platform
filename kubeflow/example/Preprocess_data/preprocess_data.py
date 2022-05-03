import json
import logging
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)

def _preprocessing_data(args):
    
    logging.info('Loading data:' +str(args.input_path))
    data = pd.read_csv(args.input_path, index_col=0)
    
    data.drop(['entity_id'], axis=1,
             inplace=True)
    
    logging.info('Splitting data into train and test')
    X_train, X_test, y_train, y_test = train_test_split(data.loc[:, data.columns!='target'],
                                                         data.loc[:, data.columns=='target'],
                                                         test_size=0.20, random_state=42)
    output_data = {'x_train' : X_train.to_json(),
                   'y_train' : y_train.to_json(),
                   'x_test' : X_test.to_json(),
                   'y_test' : y_test.to_json()}
    
    logging.info('Data split')
    
    output_data_json = json.dumps(output_data)
    
    logging.info('Saving data')
    
    with open(args.output_path, 'w') as out_file:
        json.dump(output_data_json, out_file)

if __name__ == '__main__':
    # Defining and parsing the command-line arguments
    parser = argparse.ArgumentParser(description='My program description')
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str)

    args = parser.parse_args()

    # Creating the directory where the output file will be created (the directory may or may not exist).
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    _preprocessing_data(args)