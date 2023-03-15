import numpy as np
import torch
import sys
from configUnet3 import config_func_unet3
cfg = config_func_unet3(False)


def calc_f1_batch(model_output, target_var):
    batch_length = target_var.shape[0]
    scores = []
    sigmoid_function = torch.nn.Sigmoid()
    model_output = sigmoid_function(model_output)
    for i in range(batch_length):
        thresholded = model_output[i, :, :, :] > 0.5
        thresholded_tmp = thresholded.byte()
        picturex = torch.squeeze(thresholded_tmp)
        picturex = picturex.detach().numpy()
        picturex = picturex.astype(float)
        pic_results = calc_f1_picture(picturex, target_var[i])
        scores.append(pic_results)

    return scores

def calc_f1_picture(output, target):
    target_pic = torch.squeeze(target)
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for x in range(output.shape[0]):
        for y in range(output.shape[1]):
            #false positives kad je predikcija 1 a target 0
            if output[x][y] == 1:
                if target_pic[x][y] == 0:
                    false_positives += 1
            #false negatives kad je predikcija 0 a target je 1
            if output[x][y] == 0:
                if target_pic[x][y] == 1:
                    false_negatives += 1
            #total positives
            if target_pic[x][y] == 1:
                true_positives += 1
    #Division by zero handling
    if true_positives + false_positives == 0:
        accuracy = 1.0
    else:
        accuracy = true_positives / (true_positives + false_positives)
    if true_positives + false_negatives == 0:
        recall = 1.0
    else:
        recall = true_positives / (true_positives + false_negatives)
    return accuracy, recall