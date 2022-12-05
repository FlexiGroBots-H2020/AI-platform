import sys
import pandas as pd
from os import listdir
from os.path import isfile, join
import argparse
from pathlib import Path
from minioConector import minioConector
import datetime


#pipeline purpose only




#path = '/home/jovyan/vol-1/logs/Train_BGFG_BCE_with_weightsUnet3/'
parser = argparse.ArgumentParser(description='My program description')
parser.add_argument('--new_location', type=str)
parser.add_argument('--log_location', type=str)
args = parser.parse_args()



#path = '/mnt/logs/Train_BGFG_BCE_with_weightsUnet3/'
path = args.new_location

sys.path.append(path)
print(sys.path)

onlydirs = [f for f in listdir(path)]

best_result = -1
best_result_location = None
result_location = None
tmp_file = ''
#rezultati = pd.DataFrame(columns = ['Background', 'Borovnica'])

rezultati = pd.DataFrame(columns = ['Model', 'Architecture','Learning rate', 'Lambda', 'Step', 'Batch size','Background', 'Borovnica'])

for d in onlydirs:
    tmp_file = None
    result_location = None
    result_list = listdir(path + d)
    for r in result_list:
        if r[-3:] == 'csv':
            tmp_file = r

        if r[-2:] == 'pt':
            result_location = path + d + "/" + r

    parse_name = tmp_file.split('_')

    tmp_batch_size = tmp_file[-5:-4]

    tmp_model = tmp_file[:3]

    #print(tmp_file)

    tmp_lr = parse_name[2]

    tmp_step = parse_name[4]

    tmp_Lambda = parse_name[7]

    tmp_loss = parse_name[10]

    tmp_arhitektura = parse_name[12]

    result_file = path + d + '/' + tmp_file
    result_csv = pd.read_csv(result_file)
    #print(type(result_csv['background']))
    #rezultati = rezultati.append({'Background': result_csv['background'][0], 'Borovnica':result_csv['Borovnica'][0]}, ignore_index=True)
    rezultati.loc[len(rezultati.index)] = [tmp_model, tmp_arhitektura, tmp_lr, tmp_Lambda, tmp_step, tmp_batch_size, result_csv['background'][0], result_csv['Borovnica'][0]]

    if float(result_csv['Borovnica'][0]) > best_result:
        best_result = result_csv['Borovnica'][0]
        print(result_location)
        best_result_location = result_location


rezultati = rezultati.sort_values('Borovnica', ascending=False).reset_index()
rezultati = rezultati.drop('index', axis=1)

print("best result")
print(best_result)
print("best result location")
print(best_result_location)

if best_result_location != None:

    x = datetime.datetime.now()
    x = str(x).replace(' ',';')
    rezultati.to_csv('/mnt/results/res-' + x + '.csv')


    Minio_object= minioConector()
    Minio_object.uploadFiles("blueberry-results","res-" + x + ".csv",'/mnt/results/res-' + x + '.csv')
    Minio_object.uploadFiles("blueberry-results","best_model_" + x + ".pt", best_result_location)
else:
    print("No result")
