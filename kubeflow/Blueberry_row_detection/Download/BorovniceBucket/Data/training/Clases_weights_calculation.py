from turtle import pos
import matplotlib.pyplot as plt
import cv2 as cv
import copy
import pickle
import torch.utils.data.dataloader
import cv2
from sklearn.utils import class_weight
from torch import nn
from torch.nn import functional as F
import gc
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import torch
from torchvision.ops.boxes import _box_inter_union
import os, sys
from torch.nn import BCEWithLogitsLoss
import random
from datetime import datetime
import numpy as np
from torchsummary import summary
# from Fully_Convolutional_Net_Torch import *
# import timm
from Test_Overfit_utils import *
from torchsummary import summary
# from Unet_hayashimasa import UNet as UNet_hayashimasa
# from Unet_Milesial import UNet as UNet_Milesial
# from Unet_Timm import Unet as UNet_Timm
# from Boundry_Loss import BoundaryLoss
from skimage.filters import threshold_otsu
from sklearn.metrics import precision_recall_curve, auc, f1_score
from skimage.io import imread, imsave
import glob, os
from skimage import io
from skimage import color
from skimage import segmentation


# # URL for tiger image from Berkeley Segmentation Data Set BSDS
# # url=('http://www.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300/html/images/plain/normal/color/108073.jpg')
# url=('ZAD3NZMXK_5223-878-5735-1390_rgb.jpg')
#
# # Load tiger image from URL
# tiger = io.imread(url)
#
# # Segment image with SLIC - Simple Linear Iterative Clustering
# seg = segmentation.slic(tiger, n_segments=4, compactness=40.0, enforce_connectivity=True, sigma=3)
#
# # Generate automatic colouring from classification labels
# io.imshow(color.label2rgb(seg,tiger))
# plt.show()


# if __name__ == '__main__':
def main(lr, raspored_lr, losovi):
    # torch.use_deterministic_algorithms(True)
    # set_random_seed = 15
    #
    # base_folder_path = os.getcwd()
    # base_folder_path = base_folder_path.replace("\\", "/")

    # lab_path = r"/home/antic/PYTHON_projekti/proj_Agrovision_Torch/UG6H862FY_2777-3999-3289-4511_label.png"
    # img_path = r"/home/antic/PYTHON_projekti/proj_Agrovision_Torch/UG6H862FY_2777-3999-3289-4511_rgb.jpg"

    # lab_path = r"D:\BIOSENS_programi\PYTHON_projekti\Projekat_Agrovision_Torch\UG6H862FY_2777-3999-3289-4511_label.png"
    # img_path_rgb = r"D:\BIOSENS_programi\PYTHON_projekti\Projekat_Agrovision_Torch\UG6H862FY_2777-3999-3289-4511_rgb.jpg"
    # img_path_nir = r"D:\BIOSENS_programi\PYTHON_projekti\Projekat_Agrovision_Torch\UG6H862FY_2777-3999-3289-4511_nir.jpg"

    # Labela ZAD3NZMXK_5223-878-5735-1390.png ; 12528 index

    classes_rgb = ['cloud_shadow', 'double_plant', 'planter_skip', 'standing_water', 'waterway', 'weed_cluster']

    # classes_rgb = ['double_plant', 'drydown', 'endrow', 'nutrient_deficiency', 'planter_skip',
                #    'storm_damage', 'water', 'waterway', 'weed_cluster']

    # labels_path = r"/storage/home/antica/DATASETS/Agriculture_Vision_2020_mini/numpy_new" ## mini dataset
    labels_path = r"/storage/home/antica/DATASETS/Agriculture_Vision_2020/train/numpy_new" ## full dataset

    dataset, _ = os.path.split(labels_path)
    dataset, _ = os.path.split(dataset)
    _, dataset = os.path.split(dataset)
    lab_data_type = '*.npy'
    img_data_type = "*.npy"
    # double plant ima oblik, planter skip je bukvalno jedan piksel sa desne strane,waterway je skroz dole, gore levo je weed cluster

    device = 'cpu'

    # sheet_name = "Sheet1"
    # labels_xlsx_classes_path = r"D:\BIOSENS_programi\PYTHON_projekti\Projekat_Agrovision_Torch\Agrovision_Classes_2020\labels"
    # labels_classes_xlsx = glob.glob(os.path.join(labels_xlsx_classes_path, '*.xlsx'))
    #
    # nir_xlsx_classes_path = r"D:\BIOSENS_programi\PYTHON_projekti\Projekat_Agrovision_Torch\Agrovision_Classes_2020\nir"
    # nir_classes_xlsx = glob.glob(os.path.join(labels_xlsx_classes_path, '*.xlsx'))
    #
    # rgb_xlsx_classes_path = r"D:\BIOSENS_programi\PYTHON_projekti\Projekat_Agrovision_Torch\Agrovision_Classes_2020\rgb"
    # rgb_classes_xlsx = glob.glob(os.path.join(labels_xlsx_classes_path, '*.xlsx'))
    #
    # loss_type = losovi  # dice or miou or bce, mse (L2), tversky, focal_tversky, lovasz, L1, L1_smooth, boundry
    #
    # net_type = "UNet3"  # efficientnet_b4 or Test_overfit or UNet or Fully_conv32,16,8 or Defalut
    # UNet; UNet_hayashimasa; UNet_Milesial ; UNet_Timm
    # UNet2,3,4,5,8 x downsample

    num_channels = 4
    num_channels_lab = 3
    img_h = 512
    img_w = 512

    files_rgb = glob.glob(os.path.join(labels_path, img_data_type))

    def load_dataset(labels_path, classes_rgb, img_data_type, label_data_type):
        # img_list_rgb = load_xls_files(rgb_classes_xlsx, sheet_name)
        # img_list_nir = load_xls_files(nir_classes_xlsx, sheet_name)
        # labels_list = load_xls_files(labels_classes_xlsx, sheet_name)
        mask_list = np.zeros([len(files_rgb),num_channels_lab-1, img_w, img_h],dtype='uint8')

        gc.collect()
        for j in range(len(files_rgb)):

            basename = os.path.basename(files_rgb[j])
            basename = basename[:-4] + lab_data_type[1:]

            mask1 = np.zeros([num_channels_lab - 1,img_w, img_h])
            
            mask1 = np.load(
                os.path.join(labels_path, basename))[4:-2, :, :]
            if num_channels_lab==3:
                mask_list[j, 0, :, :] = np.expand_dims(np.expand_dims(mask1[0,:,:],axis=0),axis=0)
                mask_list[j, 1, :, :] = np.expand_dims(mask1[1:,:,:].any(axis=0),axis=0) #np.expand_dims(mask1.any(axis=0).astype(np.uint8),axis=0)
            else:
                mask_list[j, :, :, :] = np.expand_dims(mask1,axis=0)
            print("Loaded " + str(j) + " th example ")

        return mask_list

    mask1 = load_dataset(labels_path, classes_rgb, img_data_type, lab_data_type)
    pos_weights_for_bce = np.zeros(1)
    # mask1 = mask1[:,1:]
    # negative_class = []
    # for i in range(6):
    #     negative_class = copy.deepcopy(mask1)
    #     negative_class[:,i] = 0
    #     pos_weights_for_bce[i] = np.sum(negative_class)/np.sum(mask1[:,i])
    #     # pos_weights_for_bce[i] = np.sum(mask1[:,i+1]==0)/np.sum(mask1[:,i+1]==1)
    # np.save('pos_weights_for_bce_mini_without_background.npy',pos_weights_for_bce)

    # [ 3.2682978 , 32.29673068, 13.34303044, 7.65731004, 10.65020179, 1.15231238]


#######################################################
    # print(np.bincount(np.argmax(mask1,1).ravel()))
    # class_weights = class_weight.compute_class_weight(class_weight='balanced',
    #                                                   classes=np.unique(np.argmax(mask1,1).ravel()),
    #                                                   y=np.argmax(mask1,1).ravel())
    # np.save('class_weights_mini_multiclass_with_background.npy', class_weights)
    # np.save('class_weights_full_multiclass_with_background.npy', class_weights)
    # np.save('class_weights_mini_multiclass_without_background.npy', class_weights)
    # np.save('class_weights_full_multiclass_without_background.npy', class_weights)

    class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                      classes=np.unique(np.argmax(mask1,1).ravel()),
                                                      y=np.argmax(mask1,1).ravel())
    # np.save('class_weights_mini_binary_with_background.npy', class_weights)
    np.save('class_weights_full_binary_with_background.npy', class_weights)
    np.save('class_weights_full_binary_without_background.npy', class_weights[1:])



    
    # class_weights = class_weight.compute_class_weight(class_weight='balanced',
    #                                                   classes=np.unique(mask1.ravel()),
    #                                                   y=mask1.ravel())
    # np.save('class_weights_mini_binary_with_background.npy', class_weights)

###############################################################


if __name__ == '__main__':
    # main(lr=100,net_type="UNet3")
    # lr = [0.15]
    # lr = [5,3,1,0.5,0.15,0.1,0.05,0.01,0.001]
    lr = [0.05]
    # net_type = ['UNet2','UNet3','UNet4','UNet5','UNet8']
    scheduleri = ['reducelr']
    # scheduleri = ['multiplicative']
    losovi = ['ce']
    for j in range(len(scheduleri)):
        for i in range(len(losovi)):
            for k in range(len(lr)):
                main(lr[k], scheduleri[j], losovi[i])
