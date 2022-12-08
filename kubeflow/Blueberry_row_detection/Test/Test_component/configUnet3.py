import torch
import numpy as np
import json
import pandas as pd

#######################
### Hyperparameters ###
#######################
def config_func_unet3(server=True):

    # bool
    server = server       # Flag that indicates whether to use server or local machine

    load_numpy = True   # Flag that indicates what type of data we use as input

    use_mask = False # Flag that indicates are we masking data with boundery and valid masks

    use_weights = False  # Flag that indicates whether to use class weights when initializing loss function

    do_testing = True   # Flag that indicates whether to do testing after the training is done

    count_logs_flag = False # Flag that indicates whether to plot number of classes and pixels in tensorboard, classwise and batch-wise

    zscore = False       # Flag that indicates whether we use zscore normalization in preprocessing

    binary = True       # Flag that indicates whether we do binary semantic segmentation

    freeze_backbone_weights = False # Flag that indicates whether to freeze backbone weights

    early_stop = False  # Initial early stopping flag

    save_best_model = True # Initial "best model" flag, indicates whether to save model in corresponding epoch


    # strings

    scheduler_lr = 'multiplicative' # Indicates which scheduler to use

    dataset = "Borovnica"            # "mini" or "full" Indicates whether we want to use full dataset for training, validation and testing or decimated version
    # year = "2020" # "2020" or "2021"
    if server:                  # Depending on server flag, we use different device settings:
            device = "cuda"     # if server is True, that is, if we are using server machine, device will be set as "cuda"
    elif server == False:
        device = "cpu"          # else if server is False and we are using local machine or server access node, device will be set as "cpu"

    if dataset == 'Proba':  # paths to the datasets
        numpy_path = r"/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/proba/trening_set/img"
        numpy_valid_path = r"/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/proba/validation_set/img"
        numpy_test_path = r"/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/proba/test_set/img"
    elif dataset == 'Borovnica':   
        numpy_path = r"/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/leave_one_out/Experiment5/trening_set_masked/img"              # mini train dataset
        numpy_valid_path = r"/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/leave_one_out/Experiment5/validation_set/img"  # mini validation dataset
        numpy_test_path = r"/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/leave_one_out/Experiment5/test_set/img"    # mini test dataset

    classes_labels = ['Borovnica']  # Classes that we are trying to detect. For BCE, background class is rejected 
    classes_labels2 = ['background','Borovnica'] # classes_labels + background


    tb_img_list = [ ########## Train samples which we want to visualize in tensorboard during the training 
                        'svi_kanali_combo_8',#'svi_kanali_combo_8',
                        # '', '',
                        # '','',
                        # '', '',
                    ########## Validation samples which we want to visualize in tensorboard during the training
                    'svi_kanali_combo_8',#''
                    #    ,'',
                    #    '', ''
                    ]

    loss_type = 'bce'       # Indicates loss type we want to use: bce, ce, ce_1

    net_type = "UNet3"      # Indicates Architecture that we want to use: UNet3, Unet_orig,UNet3_modified...

    img_data_format = '.npy'  # Indicated the type of data we use as input

    # Integer

    # epochs = epochs   # Number of epochs  we want our model training 

    set_random_seed = 15    # Setting random seed for torch random generator

    batch_size = 4      # Size of batch during the training,validation and testing

    shuffle_state = 1   # Random shuffle seeed

    GPU_list = [0]      # Indices of GPUs that we want to allocate and use during the training

    weight_decay = 0    # L2 penalty  

    optimizer_patience = 3 #  When using ReduceLR scheduler: Indicates number of epochs without loss minimum decrease,
        # after which the learning rate will be multiplied by a lambda parameter


    save_checkpoint_freq = 1    #Initial frequency parameter for model saving

    if save_checkpoint_freq < 10:  # Further updating saving frequency parameter
        save_checkpoint_freq = 0
    if save_checkpoint_freq >= 100:
        save_checkpoint_freq = int(epochs / 10)  # frequency of saving checkpoints in epochs
    if save_checkpoint_freq == 0:
        save_checkpoint_freq = 1000

    num_channels = 5    # Number of input channels: Red, Green, Blue, NIR, Red-Edge

    num_channels_lab = len(classes_labels)

    img_h = 512 # Input image Height

    img_w = 512 # Input image Weight

    img_size = [img_h, img_w] # Input channel and label shape

    #######################
    ### loss containers ###
    #######################
    train_losses = []   # Container in which we store losses for each training batch

    validation_losses = [] # Container in which we store losses for each validation batch
    #######################

    ################
    ### Counters ###
    ################
    epoch_model_last_save = 0 # Counter that counts number of epochs since the most recent model saving

    count_train = 0      # Counter that counts number of training batches

    count_val = 0        # Counter that counts number of validation batches

    count_train_tb = 0  

    ######################
    ### early stopping ###
    ######################

    es_min = 1e9    # Initial minimum parameter for early stopping

    es_epoch_count = 0  # Epoch counter for early stopping

    es_check = 5 # Number of epochs after wich we dont have new minimal validation loss and after wich we apply early stopping

    # Dictionary creation

    dictionary = {

        "save_checkpoint_freq" : save_checkpoint_freq,
        "set_random_seed" : set_random_seed,
        #"epochs" : epochs,
        "use_mask" : use_mask,
        "count_train" : count_train,
        "count_train_tb" : count_train_tb,
        "count_val" : count_val,
        "GPU_list" : GPU_list,
        "shuffle_state" : shuffle_state,
        "weight_decay" : weight_decay,
        #"batch_size" : batch_size,
        "es_min" : es_min,
        "es_check" : es_check,
        "es_epoch_count" : es_epoch_count,
        "optimizer_patience" : optimizer_patience,
        "num_channels" : num_channels,
        "num_channels_lab" : num_channels_lab,
        "img_h" : img_h,
        "img_w" : img_w,
        "img_size": img_size,
        "epoch_model_last_save" : epoch_model_last_save,
        "scheduler_lr" : scheduler_lr,
        "dataset" : dataset,
        "tb_img_list" : tb_img_list,
        "server" : server,
        "img_data_format" : img_data_format,
        "net_type" : net_type,
        "device":device,
        # "loss_type" : loss_type,
        # "numpy_path": numpy_path,
        # "numpy_valid_path": numpy_valid_path,
        # "numpy_test_path": numpy_test_path,
        "load_numpy" : load_numpy,
        "use_weights" : use_weights, # for CE loss
        "do_testing" : do_testing,
        "count_logs_flag" : count_logs_flag,
        "freeze_backbone_weights" : freeze_backbone_weights,    
        "zscore" : zscore,
        "binary" : binary,
        "early_stop_flag" : early_stop,
        "save_best_model" : save_best_model,
        "train_losses" : train_losses,
        "validation_losses" : validation_losses,
        "classes_labels" : classes_labels,
        "classes_labels2" : classes_labels2,
    }
    
    # Serializing json 
    json_object = json.dumps(dictionary, indent = 4)
    
    # Writing to sample.json
    with open("config_Unet3.json", "w") as outfile:
        outfile.write(json_object)

    # legend_path = r"/storage/home/antica/PYTHON_projekti/Agrovision_Torch/Legend_Classes.png"
    if loss_type == 'bce':
        background_flag = False
    else:
        background_flag = True
    background_names = []
    background_area = []
    foreground_names = []
    foreground_area = [] 
    test_losses = []
    iou_per_test_image_fg = []
    k_index = 1
    dictionary_test = {

        "test_losses" : test_losses,
        "background_names" : background_names,
        "foreground_names" : foreground_names,
        "background_area" : background_area,
        "foreground_area" : foreground_area,
        "iou_per_test_image_fg" : iou_per_test_image_fg,
        "k_index" : k_index,
        "binary" : binary,
        "use_mask" : use_mask,
        "dataset" : dataset,
        "background_flag" : background_flag, 
    }
    # Serializing json 
    json_object = json.dumps(dictionary_test, indent = 4)
    
    # Writing to sample.json
    with open("config_test_Unet3.json", "w") as outfile:
        outfile.write(json_object)
