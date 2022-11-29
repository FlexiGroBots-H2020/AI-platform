from tabnanny import check
import numpy as np
from augmentation_script import augmentation_and_saving
import matplotlib.pyplot as plt
import torch
# from model import UNET
from Unet_LtS import UNet3, UNet3_modified
import sys, os
import segmentation_models_pytorch as smp
def iou_coef(y_true,y_pred):
    
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = torch.sum(y_true_f * y_pred_f)
    union = torch.sum(y_true_f) + torch.sum(y_pred_f) - intersection
    smooth = 0.0001
    # smooth = 0
    return (intersection + smooth) / (union + smooth)

def main(training_flag,validation_flag,testing_flag,png_flag):
    if training_flag:
    ### trening

        trening_labele = np.load(r'/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/leave_one_out/Experiment5/labele npy/trening_labele_croped_leave_parcel_out_exp5.npy')
        red= np.load(r'/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/leave_one_out/Experiment5/kanali npy/trening_ch_red_masked_croped.npy')
        red = (red-np.min(red))/(np.max(red)-np.min(red))*255
        green= np.load(r'/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/leave_one_out/Experiment5/kanali npy/trening_ch_green_masked_croped.npy')
        green = (green-np.min(green))/(np.max(green)-np.min(green))*255
        blue= np.load(r'/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/leave_one_out/Experiment5/kanali npy/trening_ch_blue_masked_croped.npy')
        blue = (blue-np.min(blue))/(np.max(blue)-np.min(blue))*255
        rgb = np.stack([red,green,blue],axis=2)
        nir = np.load(r'/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/leave_one_out/Experiment5/kanali npy/trening_ch_nir_masked_croped.npy')
        nir = (nir-np.min(nir))/(np.max(nir)-np.min(nir))*255
        rededge = np.load(r'/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/leave_one_out/Experiment5/kanali npy/trening_ch_rededge_masked_croped.npy')
        rededge = (rededge-np.min(rededge))/(np.max(rededge)-np.min(rededge))*255
    elif validation_flag:
    ### validacija

        val_labele = np.load(r'/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/leave_one_out/Experiment5/labele npy/validacione_labele_croped_leave_parcel_out_exp5.npy')
        red= np.load(r'/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/leave_one_out/Experiment5/kanali npy/validacioni_ch_red_croped.npy')
        red = (red-np.min(red))/(np.max(red)-np.min(red))*255
        green= np.load(r'/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/leave_one_out/Experiment5/kanali npy/validacioni_ch_green_croped.npy')
        green = (green-np.min(green))/(np.max(green)-np.min(green))*255
        blue= np.load(r'/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/leave_one_out/Experiment5/kanali npy/validacioni_ch_blue_croped.npy')
        blue = (blue-np.min(blue))/(np.max(blue)-np.min(blue))*255
        rgb = np.stack([red,green,blue],axis=2)
        nir = np.load(r'/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/leave_one_out/Experiment5/kanali npy/validacioni_ch_nir_croped.npy')
        nir = (nir-np.min(nir))/(np.max(nir)-np.min(nir))*255
        rededge = np.load(r'/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/leave_one_out/Experiment5/kanali npy/validacioni_ch_red_edge_croped.npy')
        rededge = (rededge-np.min(rededge))/(np.max(rededge)-np.min(rededge))*255
    elif testing_flag:
    ### test
    #/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica
        test_labele = np.load(r'/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/leave_one_out/Experiment5/labele npy/test_labele_croped_leave_parcel_out_exp5.npy')
        red= np.load(r'/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/leave_one_out/Experiment5/kanali npy/test_ch_red_masked_croped.npy')
        red = (red-np.min(red))/(np.max(red)-np.min(red))*255
        green= np.load(r'/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/leave_one_out/Experiment5/kanali npy/test_ch_green_masked_croped.npy')
        green = (green-np.min(green))/(np.max(green)-np.min(green))*255
        blue= np.load(r'/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/leave_one_out/Experiment5/kanali npy/test_ch_blue_masked_croped.npy')
        blue = (blue-np.min(blue))/(np.max(blue)-np.min(blue))*255
        rgb = np.stack([red,green,blue],axis=2)
        nir = np.load(r'/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/leave_one_out/Experiment5/kanali npy/test_ch_nir_masked_croped.npy')
        nir = (nir-np.min(nir))/(np.max(nir)-np.min(nir))*255
        rededge = np.load(r'/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/leave_one_out/Experiment5/kanali npy/test_ch_rededge_masked_croped.npy')
        rededge = (rededge-np.min(rededge))/(np.max(rededge)-np.min(rededge))*255
    else:
        print('Error: wrong flags')
        sys.exit(0)
    def load_checkpoint(checkpoint,model):
        print("=> Loading checkpoint")
        try:
            model.load_state_dict(checkpoint["state_dict"])
        except:
            try:
                model.load_state_dict(checkpoint)
            except:
                from collections import OrderedDict
                new_checkpoint = OrderedDict()
                for k, v in checkpoint.items():
                    name = k[7:]
                    new_checkpoint[name]= v

                for key in new_checkpoint:
                    print(key)
                
                model.load_state_dict(new_checkpoint)
    net_type = 'Unet++' # 'YT', 'GVI', 'UnetDimitrije', 'Unet++
    DEVICE = 'cpu'
    H = np.shape(rededge)[0]
    W = np.shape(rededge)[1]
    Predict_Final_Image=  np.zeros(shape=(H,W))
    tmp_labels = np.zeros([512,512])
    tmp_rgb = np.zeros([512,512,3])
    tmp_nir = np.zeros([512,512])
    tmp_red_edge = np.zeros([512,512])
    tmp_svi_kanali_combo = np.zeros([512,512])
    miou = []
    counter = 0
    if net_type == 'UnetDimitrije':
        model = UNet3(n_channels=5, n_classes=1, height=H, width= W, zscore = False)
    elif net_type == 'Unet++':
        model = smp.UnetPlusPlus(in_channels=5, encoder_depth=3, classes=1,activation=None,decoder_channels=[64, 32, 16]).to(device=DEVICE)
        model = torch.nn.DataParallel(model, device_ids=[0]).to(DEVICE)

    elif net_type == 'GVI':
        model = UNet3_modified( n_channels = 5,  n_classes = 1, height = H, width = W, no_indices = 50)
    # elif net_type == "YT":
    #     model = UNET(in_channels=5, out_channels=1).to(device=DEVICE)

    predict = False
    if predict == True:
        if net_type == 'YT':
            load_checkpoint(torch.load(r"/home/stefanovicd/DeepSleep/Borovnice/UNET_YT_VERSION/my_checkpoint39_unmasked_lr_0.001.pth.tar"),
                                model)
        else:
            # load_checkpoint(torch.load(r"./logs/Train_BGFG_BCE_with_weights/0_10_06_2022_15_49_44_lr_0.0001_step_na_5_epoha_lambda_parametar_1_batch_size_8_sched_multiplicative_loss_bce/NN_model_ep_40_Train_BGFG_BCE_with_weights/fully_trained_model_epochs_39.pt",map_location=torch.device(DEVICE)),model)
            # load_checkpoint(torch.load(r"./logs/Train_BGFG_BCE_with_weights/0_10_06_2022_15_49_54_lr_0.001_step_na_5_epoha_lambda_parametar_1_batch_size_8_sched_multiplicative_loss_bce/NN_model_ep_40_Train_BGFG_BCE_with_weights/fully_trained_model_epochs_39.pt",map_location=torch.device(DEVICE)),model)
            load_checkpoint(torch.load(r"./logs/Train_BGFG_BCE_with_weights/48_09_07_2022_13_40_30_lr_1e-05_step_na_5_epoha_lambda_parametar_1_batch_size_4_sched_multiplicative_loss_bce/NN_model_ep_40_Train_BGFG_BCE_with_weights/fully_trained_model_epochs_39.pt",map_location={"cuda:0" : "cpu"}),model)
    counter1 = 0
    h_idx = 0
    w_idx = 0
    for i in range(H//512):
        w_idx = 0
        for j in range(W//512):
            if training_flag:
                tmp_labels = trening_labele[i*512:i*512+512,j*512:j*512+512]
            elif validation_flag:
                tmp_labels = val_labele[i*512:i*512+512,j*512:j*512+512]
            else:
                # tmp_labels = test_labele[i*512:i*512+512,j*512:j*512+512]
                tmp_labels = test_labele[h_idx:h_idx+512,w_idx:w_idx+512]
                
            # tmp_rgb = rgb[i*512:i*512+512,j*512:j*512+512]
            # tmp_nir = nir[i * 512:i * 512 + 512, j * 512:j * 512 + 512]
            # tmp_nir = np.expand_dims(tmp_nir, axis=2)
            # tmp_red_edge = rededge[i * 512:i * 512 + 512, j * 512:j * 512 + 512]

            tmp_rgb = rgb[h_idx:h_idx+512,w_idx:w_idx+512]
            tmp_nir = nir[h_idx:h_idx+512,w_idx:w_idx+512]
            tmp_nir = np.expand_dims(tmp_nir, axis=2)
            tmp_red_edge = rededge[h_idx:h_idx+512,w_idx:w_idx+512]

            tmp_red_edge = np.expand_dims(tmp_red_edge, axis=2)
            tmp_svi_kanali_combo = np.concatenate([tmp_rgb,tmp_nir,tmp_red_edge],axis=2)
            
            if np.sum(tmp_labels)!= 0:
                
                if predict == True:
                    if tmp_labels.shape == (512,512):
                        if net_type == 'YT':
                            x = torch.tensor(tmp_svi_kanali_combo,device = 'cuda',dtype=torch.float).permute(2, 0, 1).unsqueeze(0)
                        else:
                            x = torch.tensor(tmp_svi_kanali_combo,device = 'cpu',dtype=torch.float).permute(2, 0, 1).unsqueeze(0)
                        sigmoid_func = torch.nn.Sigmoid()
                        print(x)
                        print(model(x))
                        preds = sigmoid_func(model(x))
                        preds = (preds > 0.5).byte()
                        # plt.figure(),plt.imshow(preds.detach().cpu().numpy()[0,0])
                        # plt.figure(),plt.imshow(tmp_labels)
                        
                        
                        iou = iou_coef(torch.Tensor(tmp_labels),preds[0,0])
                        
                        if iou < 0.5:
                            counter1 += 1
                            print('kraj')
                        miou.append(iou)
                        
                        Predict_Final_Image[h_idx:h_idx+512,w_idx:w_idx+512]+=preds.detach().cpu().numpy()[0,0]

                    else:
                        continue
                else:
                    if training_flag:
                        
                        if png_flag:
                            path_img = r'/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/PNG/trening_set_masked/img/svi_kanali_combo_' + str(
                            counter)
                            path_label = r'/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/PNG/trening_set_masked/label/svi_kanali_combo_'+str(
                            counter)
                            augmentation_and_saving(tmp_svi_kanali_combo, tmp_labels, path_img, path_label, testing_flag, png_flag)

                        else:
                            path_img = r'/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/leave_one_out/Experiment5/trening_set_masked/img/svi_kanali_combo_' + str(
                            counter)
                            path_label = r'/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/leave_one_out/Experiment5/trening_set_masked/label/svi_kanali_combo_'+str(
                            counter)
                            augmentation_and_saving(tmp_svi_kanali_combo, tmp_labels, path_img, path_label, testing_flag, png_flag)

                    elif validation_flag:
                        
                        if png_flag:
                            path_img = r'/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/PNG/validation_set/img/svi_kanali_combo_' + str(
                            counter)
                            path_label = r'/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/PNG/validation_set/label/svi_kanali_combo_' + str(
                            counter)
                            augmentation_and_saving(tmp_svi_kanali_combo, tmp_labels, path_img, path_label, testing_flag, png_flag)

                        else:
                            path_img = r'/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/leave_one_out/Experiment5/validation_set/img/svi_kanali_combo_' + str(
                            counter)
                            path_label = r'/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/leave_one_out/Experiment5/validation_set/label/svi_kanali_combo_' + str(
                            counter)
                            augmentation_and_saving(tmp_svi_kanali_combo, tmp_labels, path_img, path_label, testing_flag, png_flag)

                    elif testing_flag:
                        
                        if png_flag:
                            path_img = r'/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/PNG/test_set2/img/svi_kanali_combo_' + str(
                            counter)
                            path_label = r'/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/PNG/test_set2/label/svi_kanali_combo_' + str(
                            counter)
                            augmentation_and_saving(tmp_svi_kanali_combo, tmp_labels, path_img, path_label, testing_flag, png_flag)
                        else:
                            path_img = r'/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/leave_one_out/Experiment5/test_set/img/svi_kanali_combo_' + str(
                            counter)
                            path_label = r'/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/leave_one_out/Experiment5/test_set/label/svi_kanali_combo_' + str(
                            counter)
                            augmentation_and_saving(tmp_svi_kanali_combo, tmp_labels, path_img, path_label, testing_flag, png_flag)

                    

                print('sample '+str(counter))
                counter+=1
            w_idx+=512
        h_idx+=512
    # plt.figure(),plt.imshow(Predict_Final_Image)
    # plt.imsave("PREDICTION_RESULT_th05_Unet++_"+ net_type +"_40epoha_Zaragosa_lr1e-5.png",Predict_Final_Image)
    # np.save("PREDICTION_RESULT_th05_Unet3++_"+ net_type +"_40epoha_Zaragosa_lr1e-5.npy",Predict_Final_Image)
    # np.save("miou_test_"+net_type+"_Zaragosa",miou)
    print('kraj')



if __name__ == "__main__":
    
    main(training_flag = True,validation_flag = False, testing_flag = False, png_flag = False)
    # main(False,True,False,False)
    # main(False,False,True,False)
