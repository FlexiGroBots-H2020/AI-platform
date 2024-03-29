from calendar import EPOCH
import matplotlib.pyplot as plt
from numpy import binary_repr
import torch.utils.data.dataloader
from torch.utils.tensorboard import SummaryWriter
import random
from torchsummary import summary
import os
from print_utils import *
from data_utils import *
from loss_utils import *
from model_utils import *
from tb_utils import *
from metrics_utils import*
from configUnet3 import config_func_unet3
import argparse
# from focal_loss import FocalLoss2
import time

def upisivanje(ispis, ime_foldera):
    fff = open(ime_foldera, "a")
    fff.write(str(ispis) + "\n")
    fff.close()

def set_seed(seed):
    torch_manual_seed = torch.manual_seed(seed)
    torch_manual_seed_cuda = torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)
    torch.use_deterministic_algorithms(False)
    torch.random.seed()
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    return torch_manual_seed, torch_manual_seed_cuda

def main(putanja_train, putanja_val, putanja_test, p_index,lr,lambda_p,step, num_epochs, loss_type,Batch_size):


    # lr = 1e-5
    # lambda_parametri = 1
    # stepovi = 5


    lr = lr
    lambda_parametri = lambda_p
    stepovi = step
    epochs = num_epochs
    batch_size = Batch_size


    tmp = get_args('train','UNet3')
    globals().update(tmp)
    # print(device)
    base_folder_path = os.getcwd()
    base_folder_path = base_folder_path.replace("\\", "/")

    ime_foldera_za_upis,logs_path, save_model_path = pretraining_prints(p_index,lr,stepovi,lambda_parametri,batch_size,loss_type,net_type,epochs)

    ####################
    ### data loading ###
    ####################

    train_loader, valid_loader = data_loading(ime_foldera_za_upis,putanja_train,putanja_val,binary,p_index,net_type,batch_size)

    ####################
    after_data_loading_prints(lr,ime_foldera_za_upis,train_loader,valid_loader,batch_size,epochs,loss_type)
    ####################

    torch_manual_seed, torch_manual_seed_cuda = set_seed(set_random_seed)

    tb = SummaryWriter(log_dir=logs_path)
    ############################
    ### model initialization ###
    ############################
    # print(device)
    segmentation_net = model_init(num_channels,num_channels_lab,img_h,img_w,zscore,net_type,device,server,GPU_list)
    segmentation_net = torch.nn.DataParallel(segmentation_net, device_ids=[0]).to(device)

    # print(summary(segmentation_net,(5,512,512)))
    ############################
    ### model initialization ###
    ############################

    optimizer, scheduler = optimizer_init(segmentation_net,lr,weight_decay,scheduler_lr,lambda_parametri,optimizer_patience)

    ############################
    ### Loss initialization ###
    ############################

    criterion = loss_init(use_weights,loss_type,dataset,num_channels_lab,device,use_mask)

    if server:
        start_train = torch.cuda.Event(enable_timing=True)
        start_val = torch.cuda.Event(enable_timing=True)
        end_train = torch.cuda.Event(enable_timing=True)
        end_val = torch.cuda.Event(enable_timing=True)

    # Brojanje Iteracija
    global count_train
    global count_val
    global es_min
    global epoch_model_last_save
    epoch_list = np.zeros([epochs])
    all_train_losses = np.zeros([epochs])
    all_validation_losses = np.zeros([epochs])
    all_lr =np.zeros([epochs])
    val_loss_es = torch.zeros(epochs)

    for epoch in range(epochs):

        train_part = "Train"
        segmentation_net.train(mode=True)
        print("Epoch %d: Train[" % epoch, end="")

        if server:
            start_train.record()
            torch.cuda.empty_cache()
        else:
            start_train = time.time()
        index_start = 0

        batch_iou = torch.zeros(size=(len(train_loader.dataset.img_names),num_channels_lab*2),device=device,dtype=torch.float32)
        print(type(loss_type))
        print(loss_type)

        loss_type = 'bce'

        if loss_type == 'bce':
            batch_iou_bg = torch.zeros(size=(len(train_loader.dataset.img_names),2),device=device,dtype=torch.float32)


        for input_var, target_var, batch_names_train in train_loader:

            set_zero_grad(segmentation_net)

            model_output = segmentation_net.forward(input_var)
            # mask_train = torch.logical_and(mask_train[:,0,:,:],mask_train[:,1,:,:])
            loss = loss_calc(loss_type,criterion,model_output,target_var,num_channels_lab,use_mask)
            loss.backward()

            optimizer.step()  # mnozi sa grad i menja weightove

            train_losses.append(loss.data)

            ######## update!!!!

            index_end = index_start + len(batch_names_train)
            if loss_type == 'bce':
                batch_iou[index_start:index_end, :],batch_iou_bg[index_start:index_end]= calc_metrics_pix(model_output, target_var, num_channels_lab,device,use_mask,loss_type)
            elif loss_type == 'ce':
                batch_iou[index_start:index_end, :]= calc_metrics_pix(model_output, target_var,mask_train, num_channels_lab, device, use_mask,loss_type)
            else:
                print("Error: unimplemented loss type")
                sys.exit(0)
            index_start += len(batch_names_train)
            ###########################################################
            ### iscrtavanje broja klasa i broja piskela i tako toga ###
            ###########################################################

            if epoch == 0 and count_logs_flag:
                count_freq = 2
                tb_num_pix_num_classes(tb, count_train, count_train_tb, count_freq, num_channels_lab,\
                    batch_names_train, target_var, classes_labels, loss)

            #########################################################################
            ### Iscrtavanje trening uzoraka sa predefinisane liste u tensorboard  ###
            #########################################################################

            tb_image_list_plotting(tb, tb_img_list, num_channels_lab, epoch, input_var, target_var,\
                 model_output, train_part, device, batch_names_train,use_mask,dataset,loss_type)

            count_train += 1
            print("*", end="")

        #########################################################
        ### Racunanje finalne metrike nad celim trening setom ###
        #########################################################
        if loss_type == 'bce':
            final_metric_calculation(tensorbd=tb,loss_type = loss_type,epoch=epoch,num_channels_lab=num_channels_lab,classes_labels = classes_labels,\
                batch_iou_bg=batch_iou_bg,batch_iou=batch_iou,train_part= train_part,ime_foldera_za_upis=ime_foldera_za_upis)
        elif loss_type == 'ce':
            final_metric_calculation(tensorbd=tb,loss_type = loss_type,epoch=epoch,num_channels_lab=num_channels_lab,classes_labels = classes_labels,\
                batch_iou=batch_iou,train_part= train_part,ime_foldera_za_upis=ime_foldera_za_upis)
        else:
            print("Error: Unimplemented loss type!")
            sys.exit(0)

        if server:
            end_train.record()
            torch.cuda.synchronize()
            print("] ")
            ispis = ("Time Elapsed For Train epoch " + str(epoch) + " " + str(start_train.elapsed_time(end_train) / 1000))
            print(ispis)
            upisivanje(ispis, ime_foldera_za_upis)
        else:
            end_train = time.time()
            ispis = ("Time Elapsed For Train epoch " + str(epoch) + " " + str(end_train-start_train))
            print(ispis)
            upisivanje(ispis, ime_foldera_za_upis)
        all_train_losses[epoch] = (torch.mean(torch.tensor(train_losses,dtype = torch.float32)))
        all_lr[epoch] = (optimizer.param_groups[0]['lr'])


        if epoch !=0 and (epoch % stepovi)==0 :
            print("epoha: " + str(epoch) +" , uradjen step!")


        print(" Validation[", end="")
        del train_part

        if server:
            torch.cuda.empty_cache()
        train_part = "Valid"
        segmentation_net.eval()

        if server:
            start_val.record()
        else:
            start_val = time.time()
        index_start = 0

        batch_iou = torch.zeros(size=(len(valid_loader.dataset.img_names),num_channels_lab*2),device=device,dtype=torch.float32)
        if loss_type == 'bce':
            batch_iou_bg = torch.zeros(size=(len(valid_loader.dataset.img_names),2),device=device,dtype=torch.float32)

        for input_var, target_var, batch_names_valid in valid_loader:

            model_output = segmentation_net.forward(input_var)
            # mask_val = torch.logical_and(mask_val[:,0,:,:],mask_val[:,1,:,:])
            val_loss = loss_calc(loss_type,criterion,model_output,target_var, num_channels_lab ,use_mask)

            validation_losses.append(val_loss.data)

            index_end = index_start + len(batch_names_valid)
            if loss_type == 'bce':
                batch_iou[index_start:index_end, :], batch_iou_bg[index_start:index_end]= calc_metrics_pix(model_output, target_var, num_channels_lab,device,use_mask,loss_type)
            elif loss_type == 'ce':
                batch_iou[index_start:index_end, :]= calc_metrics_pix(model_output, target_var, num_channels_lab, device, use_mask,loss_type)
            else:
                print("Error: unimplemented loss type")
                sys.exit(0)

            index_start += len(batch_names_valid)

            ##############################################################################
            ### iscrtavanje validacionih uzoraka sa predefinisane liste u tensorboard  ###
            ##############################################################################

            tb_image_list_plotting(tb, tb_img_list, num_channels_lab, epoch, input_var, target_var,\
                model_output, train_part, device, batch_names_valid,use_mask,dataset,loss_type)

            count_val += 1
            print("*", end="")

        index_miou = 0
        scheduler.step(torch.mean(torch.tensor(validation_losses)))
        ##############################################################
        ### Racunanje finalne metrike nad celim validacionim setom ###
        ##############################################################
        if loss_type == 'bce':
            final_metric_calculation(tensorbd=tb,loss_type = loss_type, epoch=epoch,num_channels_lab=num_channels_lab,classes_labels = classes_labels,\
                batch_iou_bg=batch_iou_bg,batch_iou=batch_iou,train_part= train_part,ime_foldera_za_upis=ime_foldera_za_upis)
        elif loss_type == 'ce':
            final_metric_calculation(tensorbd=tb,loss_type = loss_type,epoch=epoch,num_channels_lab=num_channels_lab,classes_labels = classes_labels,\
                batch_iou=batch_iou,train_part= train_part,ime_foldera_za_upis=ime_foldera_za_upis)
        else:
            print("Error: Unimplemented loss type!")
            sys.exit(0)

        if server:
            end_val.record()
            torch.cuda.synchronize()
            print("] ", end="")
            ispis = ("Time Elapsed For Valid epoch " + str(epoch) + " " + str(start_val.elapsed_time(end_val) / 1000))
            print(ispis)
            upisivanje(ispis, ime_foldera_za_upis)
        else:
            end_val = time.time()
            print("] ", end="")
            ispis = ("Time Elapsed For Valid epoch " + str(epoch) + " " + str(str(end_val-start_val)))
            print(ispis)
            upisivanje(ispis, ime_foldera_za_upis)
        epoch_list[epoch] = epoch
        all_validation_losses[epoch] = (torch.mean(torch.tensor(validation_losses,dtype = torch.float32)))

        end_of_epoch_print(epoch,all_train_losses,all_validation_losses,optimizer,ime_foldera_za_upis)

        ##############################################################
        ### ispisivanje loss vrednosti za datu epohu u tensorboard ###
        ##############################################################

        tb_add_epoch_losses(tb,train_losses,validation_losses,epoch)

        early_stop = early_stopping(epoch, val_loss_es, all_validation_losses, es_check, \
            segmentation_net, save_model_path, save_checkpoint_freq, ime_foldera_za_upis,es_min,epoch_model_last_save,es_epoch_count,save_best_model,early_stop_flag,lr,stepovi,lambda_parametri,loss_type,net_type,batch_size)
        if early_stop:
            break

    ##### upitno da li je neophodno #####
    if not(early_stop):
        fully_trained_model_saving(segmentation_net,save_model_path,epoch,ime_foldera_za_upis,lr,stepovi,lambda_parametri,loss_type,net_type,batch_size)
    #####################################
    if server:
        torch.cuda.empty_cache()

    post_training_prints(ime_foldera_za_upis)
    plt.figure(),plt.plot(all_train_losses),plt.title("Loss per epoch for UNet"),plt.plot(all_validation_losses), plt.legend(["Train loss","Validation loss"])
    plt.show()
    plt.savefig("./model_loss_training_unet_epochs_"+str(epochs)+"_lr_"+str(lr)+"_step_"+str(stepovi)+"_Lambda_parametar_"+str(lambda_parametri)+"_loss_type_"+str(loss_type)+"_arhitektura_"+str(net_type)+"_batch_size_"+str(batch_size)+".png")

    np.save(logs_path + "/all_train_losses_"+str(p_index)+str(epochs)+"_lr_"+str(lr)+"_step_"+str(stepovi)+"_Lambda_parametar_"+str(lambda_parametri)+"_loss_type_"+str(loss_type)+"_arhitektura_"+str(net_type)+"_batch_size_"+str(batch_size)+".npy", all_train_losses)
    np.save(logs_path + "/all_validation_losses_"+str(p_index)+str(epochs)+"_lr_"+str(lr)+"_step_"+str(stepovi)+"_Lambda_parametar_"+str(lambda_parametri)+"_loss_type_"+str(loss_type)+"_arhitektura_"+str(net_type)+"_batch_size_"+str(batch_size)+".npy", all_validation_losses)
    np.save(logs_path + "/all_lr.npy", all_lr)
    np.save(logs_path + "/epoch_list.npy", epoch_list)

    ###############
    ### TESTING ###
    ###############

    if do_testing:
        criterion_1 = criterion

        test_loader = AgroVisionDataLoader(img_size, putanja_test, img_data_format, p_index, # umesto p_indexa shuffle parametar stoji po difoltu ,
                                           batch_size, device, zscore,binary,dataset)
        uporedna_tabela = pd.DataFrame()
        IOU = run_testing(segmentation_net, test_loader, ime_foldera_za_upis, device, num_channels_lab, classes_labels,classes_labels2,
                     criterion_1, loss_type, tb, zscore,net_type,lr,stepovi,lambda_parametri,batch_size)

    end_prints(ime_foldera_za_upis)
    # return IOU
    # VISUALIZE TENSORBOARD
    ## tensorboard dev upload --logdir=logs/Test_Overfit
    # # tensorboard --logdir=logs/Train_BCE_with_weights --host localhost

if __name__ == '__main__':
    config_func_unet3(server=False)
    # lr = [1e-2,1e-3,1e-4]
    # trening_location = "/mnt/FullSet/trening_set_mini/img"
    # validation_location = "/mnt/FullSet/validation_set_mini/img"
    # test_location = "/mnt/FullSet/test_set_mini/img"

    parser = argparse.ArgumentParser(description='My program description')
    parser.add_argument('--learning_rate', type=str)
    parser.add_argument('--lambda_parametar', type=str)
    parser.add_argument('--stepovi_arr', type=str)
    parser.add_argument('--num_epochs', type=str)
    parser.add_argument('--loss_type', type=str)
    parser.add_argument('--Batch_size', type=str)
    parser.add_argument('--trening_location', type=str)
    parser.add_argument('--validation_location', type=str)
    parser.add_argument('--test_location', type=str)
    args = parser.parse_args()


    lr = args.learning_rate
    lambda_parametar = args.lambda_parametar
    stepovi_arr = args.stepovi_arr

    num_epochs = args.num_epochs
    loss_type = args.loss_type
    Batch_size = args.Batch_size

    trening_location = args.trening_location
    validation_location = args.validation_location
    test_location = args.test_location

    lr = lr[1:-1]
    lr = lr.split(",")
    lr = [float(n) for n in lr]

    lambda_parametar = lambda_parametar[1:-1]
    lambda_parametar = lambda_parametar.split(",")
    lambda_parametar = [int(n) for n in lambda_parametar]

    stepovi_arr = stepovi_arr[1:-1]
    stepovi_arr = stepovi_arr.split(",")
    stepovi_arr = [int(n) for n in stepovi_arr]

    num_epochs = num_epochs[1:-1]
    num_epochs = num_epochs.split(",")
    num_epochs = [int(n) for n in num_epochs]

    loss_type = loss_type[2:-2]
    # loss_type = loss_type.split(",")
    # loss_type = [n for n in loss_type]

    Batch_size = Batch_size[1:-1]
    Batch_size = Batch_size.split(",")
    Batch_size = [int(n) for n in Batch_size]


    print("---inputs----")
    print(lr)
    print(lambda_parametar)
    print(stepovi_arr)
    print(num_epochs)
    print(loss_type)
    print(Batch_size)
    print(trening_location)
    print(validation_location)
    print(test_location)
    print("-----------------")


    trening_location = args.trening_location
    validation_location = args.validation_location
    test_location = args.test_location
    # lr = [1e-3]
    # lambda_parametar = [1]
    # stepovi_arr = [5]
    # num_epochs = [10]
    # loss_type = ['bce']
    # Batch_size = [8]
    # classes_labels2 = ['background','foreground']
    # classes_labels2 = ['background','cloud_shadow','double_plant','planter_skip','standing_water','waterway','weed_cluster']
    uporedna_tabela = pd.DataFrame()
    param_ponovljivosti = 1
    loss_type = ['bce']
    for p_index in range(param_ponovljivosti): # petlja kojom ispitujemo ponovljivost istog eksperimenta, p_idex - broj trenutne iteracije
        for step_index in range(len(stepovi_arr)): # petlja kojom ispitujemo kako se trening menja za razlicite korake promene lr-a, step = broj iteracija nakon kojeg ce se odraditi scheduler.step(loss)
            for lambd_index in range(len(lambda_parametar)):  # petlja kojom ispitujemo kako se trening menja za razlicite labmda parametre kojim mnozimo lr kada dodje do ispunjavanja uslova za scheduler.step(loss)
                for lr_index in range(len(lr)): # petlja kojom ispitujemo kako se trening menja za razlicite lr-ove
                    for num_of_epochs_index in range(len(num_epochs)):
                        for loss_type_index in range(len(loss_type)):
                            for Batch_size_index in range(len(Batch_size)):
                                main(trening_location,
                                validation_location,
                                test_location,
                                p_index,lr[lr_index],lambda_parametar[lambd_index],stepovi_arr[step_index], num_epochs[num_of_epochs_index],
                                loss_type[loss_type_index],Batch_size[Batch_size_index])
        # uporedna_tabela['TestSet IoU Metric '+str(p_index)] = IOU

    # uporedna_tabela = uporedna_tabela.set_axis(classes_labels2).T
    # uporedna_tabela.to_csv("Weighted BCE without background class, mini dataset, BGFG lr 1e-3 1 epochs.csv")
