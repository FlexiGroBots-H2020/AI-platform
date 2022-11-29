import matplotlib.pyplot as plt
from torch import sigmoid
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
# from focal_loss import FocalLoss2

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

def main(putanja_train, putanja_val,p_index):
    

    lr = 1e-4
    lambda_parametri = 1
    stepovi = 5

    tmp = get_args('train')
    globals().update(tmp)
    base_folder_path = os.getcwd()
    base_folder_path = base_folder_path.replace("\\", "/")

    ime_foldera_za_upis,logs_path, save_model_path = pretraining_prints(p_index,lr,stepovi,lambda_parametri,batch_size,loss_type)

    ####################
    ### data loading ###
    ####################
    
    train_loader, valid_loader = data_loading(ime_foldera_za_upis,putanja_train,putanja_val,binary)

    ####################
    after_data_loading_prints(lr,ime_foldera_za_upis,train_loader,valid_loader)
    ####################

    torch_manual_seed, torch_manual_seed_cuda = set_seed(set_random_seed)

    tb = SummaryWriter(log_dir=logs_path)
    ############################
    ### model initialization ###
    ############################

    segmentation_net = model_init(num_channels,num_channels_lab,img_h,img_w,zscore,net_type,device,server,GPU_list)
    segmentation_net = torch.nn.DataParallel(segmentation_net, device_ids=[0]).to(device)
    
    print(summary(segmentation_net,(5,512,512)))
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
        
        index_start = 0
        
        batch_iou = torch.zeros(size=(len(train_loader.dataset.img_names),num_channels_lab*2),device=device,dtype=torch.float32)
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

        all_train_losses[epoch] = (torch.mean(torch.tensor(train_losses,dtype = torch.float32)))
        all_lr[epoch] = (optimizer.param_groups[0]['lr'])

        if epoch !=0 and (epoch % stepovi)==0 :
            print("epoha: " + str(epoch) +" , uradjen step!")
            scheduler.step()

        print(" Validation[", end="")
        del train_part
        
        if server:
            torch.cuda.empty_cache()
        train_part = "Valid"
        segmentation_net.eval()

        if server:
            start_val.record()

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

        epoch_list[epoch] = epoch
        all_validation_losses[epoch] = (torch.mean(torch.tensor(validation_losses,dtype = torch.float32)))

        end_of_epoch_print(epoch,all_train_losses,all_validation_losses,optimizer,ime_foldera_za_upis)

        ##############################################################
        ### ispisivanje loss vrednosti za datu epohu u tensorboard ###
        ##############################################################
        
        tb_add_epoch_losses(tb,train_losses,validation_losses,epoch)
        
        early_stop = early_stopping(epoch, val_loss_es, all_validation_losses, es_check, \
            segmentation_net, save_model_path, save_checkpoint_freq, ime_foldera_za_upis,es_min,epoch_model_last_save,es_epoch_count,save_best_model,early_stop_flag)
        if early_stop:
            break

    ##### upitno da li je neophodno #####
    if not(early_stop):
        fully_trained_model_saving(segmentation_net,save_model_path,epoch,ime_foldera_za_upis)
    #####################################
    if server:
        torch.cuda.empty_cache()

    post_training_prints(ime_foldera_za_upis)

    np.save(logs_path + "/all_train_losses.npy", all_train_losses)
    np.save(logs_path + "/all_lr.npy", all_lr)
    np.save(logs_path + "/epoch_list.npy", epoch_list)

    ###############
    ### TESTING ###
    ###############

    if do_testing:
        criterion_1 = criterion
        
        test_loader = AgroVisionDataLoader(img_size, numpy_test_path, img_data_format, shuffle_state,
                                           batch_size, device, zscore,binary,dataset)
        
        tmp = get_args('test')
        globals().update(tmp)

        segmentation_net.eval()

    
    
        ispis = ("_____________________________________________________________Testing Start ")
        print(ispis)
        upisivanje(ispis, ime_foldera_za_upis)

        index_start = 0
        iou_res = torch.zeros([len(test_loader.dataset.img_names), num_classes * 2])
        if loss_type == 'bce':
            iou_res_bg = torch.zeros([len(test_loader.dataset.img_names),2])
        global test_losses
        sigmoid_func = torch.nn.Sigmoid()

        for input_var, target_var, img_names_test in test_loader:
            model_output = segmentation_net(input_var)
            model_output = (sigmoid_func(model_output)>0.5).byte()
            test_loss = loss_calc(loss_type, criterion_1, model_output,target_var,num_channels_lab = num_classes,use_mask= use_mask)
        # cuvanje loss-a kroz iteracije
            test_losses.append(test_loss.data)
        
        # uporedna_tabela = pd.DataFrame()
        # IOU = run_testing(segmentation_net, test_loader, ime_foldera_za_upis, device, num_channels_lab, classes_labels,classes_labels2,
        #              criterion_1, loss_type, tb, zscore)

    end_prints(ime_foldera_za_upis)
    return test_losses,model_output
    # return IOU
    # VISUALIZE TENSORBOARD
    ## tensorboard dev upload --logdir=logs/Test_Overfit
    # # tensorboard --logdir=logs/Train_BCE_with_weights --host localhost

if __name__ == '__main__':

    # lr = [1e-2,1e-3,1e-4]
    lr = [1e-2] 
    lambda_parametar = [1]
    stepovi_arr = [5]
    # classes_labels2 = ['background','foreground']
    # classes_labels2 = ['background','cloud_shadow','double_plant','planter_skip','standing_water','waterway','weed_cluster']
    uporedna_tabela = pd.DataFrame()
    param_ponovljivosti = 1
    for p_index in range(param_ponovljivosti): # petlja kojom ispitujemo ponovljivost istog eksperimenta, p_idex - broj trenutne iteracije
        for step_index in range(len(stepovi_arr)): # petlja kojom ispitujemo kako se trening menja za razlicite korake promene lr-a, step = broj iteracija nakon kojeg ce se odraditi scheduler.step(loss)
            for lambd_index in range(len(lambda_parametar)):  # petlja kojom ispitujemo kako se trening menja za razlicite labmda parametre kojim mnozimo lr kada dodje do ispunjavanja uslova za scheduler.step(loss)
                for lr_index in range(len(lr)): # petlja kojom ispitujemo kako se trening menja za razlicite lr-ove
                    main(r"/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/proba/trening_set/img",r"/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/proba/trening_set/img",p_index)
        # uporedna_tabela['TestSet IoU Metric '+str(p_index)] = IOU
        
    # uporedna_tabela = uporedna_tabela.set_axis(classes_labels2).T                
    # uporedna_tabela.to_csv("Weighted BCE without background class, mini dataset, BGFG lr 1e-3 1 epochs.csv")
