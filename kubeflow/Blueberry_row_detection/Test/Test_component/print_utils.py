
import shutil
import os,sys
from datetime import datetime
from data_utils import get_args,upisivanje



def pretraining_prints(p_index,lr,stepovi,lambda_parametri,batch_size,loss_type,net_type,epochs):
    tmp = get_args('train',net_type)
    globals().update(tmp)
    base_folder_path = os.getcwd()
    base_folder_path = base_folder_path.replace("\\", "/")
    script_name = os.path.basename(sys.argv[0][:-3])
    today1 = datetime.now()
    today = today1.strftime("%d/%m/%Y %H:%M:%S")
    today = today.replace("/", "_")
    today = today.replace(" ", "_")
    today = today.replace(":", "_")


    logs_path = base_folder_path + "/logs/" + script_name
    if not os.path.exists(logs_path):
        os.mkdir(logs_path)
    global scheduler_lr
    # logs_path = base_folder_path + "/logs/" + script_name + "/" + today  # path to the folder that we want to save the logs for Tensorboard
    logs_path = base_folder_path + "/logs/" + script_name + "/"+ str(p_index) + "_"+ today + "_lr_"+str(lr) + "_step_na_" + str(stepovi) +"_epoha_" + "lambda_parametar_" + str(lambda_parametri) +"_batch_size_"+str(batch_size)+ "_sched_" + str(scheduler_lr) + "_loss_" + loss_type  # path to the folder that we want to save the logs for Tensorboard
    if not os.path.exists(logs_path):
        os.mkdir(logs_path)
    elif os.path.exists(logs_path):
        shutil.rmtree(logs_path)
        os.mkdir(logs_path)

    save_model_path = logs_path + "/NN_model_ep_" + str(epochs) + "_" + script_name + "/"
    if not os.path.exists(save_model_path):
        os.mkdir(save_model_path)
    ime_foldera_za_upis = logs_path + "/LOGS_results_" + script_name + ".txt "

    ispis = ("_____________Experiment5________Today's date and time D,M,Y,H,M,S:", today)
    print(ispis)
    upisivanje(ispis, ime_foldera_za_upis)
    ispis = (" Ispitivanje ponovljivosti: iteracija br. "+str(p_index))
    print(ispis)
    return ime_foldera_za_upis,logs_path,save_model_path

def post_training_prints(ime_foldera_za_upis):
    today1 = datetime.now()
    today = today1.strftime("%d/%m/%Y %H:%M:%S")
    today = today.replace("/", "_")
    today = today.replace(" ", "_")
    today = today.replace(":", "_")
    ispis = ("________________________________________Today's date and time D,M,Y,H,M,S:", today)
    print(ispis)
    upisivanje(ispis, ime_foldera_za_upis)

    ispis = ("_______________________________________________________________________________KRAJ TRENIRANJA")
    print(ispis)
    upisivanje(ispis, ime_foldera_za_upis)

def end_prints(ime_foldera_za_upis):
    today1 = datetime.now()
    today = today1.strftime("%d/%m/%Y %H:%M:%S")
    today = today.replace("/", "_")
    today = today.replace(" ", "_")
    today = today.replace(":", "_")
    ispis = ("________________________________________Today's date and time D,M,Y,H,M,S:", today)
    print(ispis)
    upisivanje(ispis, ime_foldera_za_upis)

    ispis = ("_______________________________________________________________________________KRAJ IZVRSAVANJA")
    print(ispis)
    upisivanje(ispis, ime_foldera_za_upis)

def after_data_loading_prints(lr,ime_foldera_za_upis,train_loader,valid_loader,batch_size,epochs,loss_type):

    ispis = ("Size of Train dataset : ", str(len(train_loader.dataset.img_names)))
    print(ispis)
    upisivanje(ispis, ime_foldera_za_upis)

    ispis = ("Size of Validation dataset : ", str(len(valid_loader.dataset.img_names)))
    print(ispis)
    upisivanje(ispis, ime_foldera_za_upis)

    ispis = ("Size of Test dataset : ", str(len(valid_loader.dataset.img_names)))
    print(ispis)
    upisivanje(ispis, ime_foldera_za_upis)

    ispis = ("Epochs : ", epochs)
    print(ispis)
    upisivanje(ispis, ime_foldera_za_upis)

    ispis = ("Batch Size : ", batch_size)
    print(ispis)
    upisivanje(ispis, ime_foldera_za_upis)

    ispis = ("Load Numpy : ", load_numpy)
    print(ispis)
    upisivanje(ispis, ime_foldera_za_upis)

    ispis = ("Net Type : ", net_type)
    print(ispis)
    upisivanje(ispis, ime_foldera_za_upis)

    ispis = ("Use Loss weights : ", use_weights)
    print(ispis)
    upisivanje(ispis, ime_foldera_za_upis)

    ispis = ("Scheduler : ", scheduler_lr)
    print(ispis)
    upisivanje(ispis, ime_foldera_za_upis)

    ispis = ("Loss Type : ", loss_type)
    print(ispis)
    upisivanje(ispis, ime_foldera_za_upis)
    
    ispis = ("Learning Rate : ", lr)
    print(ispis)
    upisivanje(ispis, ime_foldera_za_upis)

    ispis = ("Use weight decay : ", weight_decay)
    print(ispis)
    upisivanje(ispis, ime_foldera_za_upis)

    ispis = ("Device : ", device)
    print(ispis)
    upisivanje(ispis, ime_foldera_za_upis)

    ispis = ("Num Channels : ", num_channels)
    print(ispis)
    upisivanje(ispis, ime_foldera_za_upis)

    ispis = ("Num Channels Labele : ", num_channels_lab)
    print(ispis)
    upisivanje(ispis, ime_foldera_za_upis)
    


def end_of_epoch_print(epoch,all_train_losses,all_validation_losses,optimizer,ime_foldera_za_upis):
    ispis = "Epoch: " + str(epoch) + " ; " + "Train Loss = " + str(all_train_losses[epoch]) + \
            " Validation Loss = " + str(all_validation_losses[epoch]) + " Learning rate = " + str(
        optimizer.param_groups[0]['lr'])
    print(ispis)
    upisivanje(ispis, ime_foldera_za_upis)