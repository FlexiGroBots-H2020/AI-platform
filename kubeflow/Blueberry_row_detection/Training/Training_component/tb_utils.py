from pyexpat import model
from statistics import mode
import tensorboard
import torch
import numpy as np
from data_utils import *
from metrics_utils import *

def tb_num_pix_num_classes(tb,count_train,count_train_tb,count_freq,num_channels_lab,batch_names_train,target_var,classes_labels,loss):
    if (count_train / count_freq).is_integer():
        classes_count = np.zeros(num_channels_lab+1)
        pix_classes_count = np.zeros(num_channels_lab+1)
        for batch in range(len(batch_names_train)):
            if num_channels_lab == 1:
                tresholded = target_var[batch,:,:,:].squeeze().byte()
                classes_count +=1
                pix_classes_count[0] = torch.sum(tresholded)
                pix_classes_count[1] = torch.sum((tresholded==0).byte())
                classes = torch.unique(tresholded)
                tb.add_scalar("Count_Train/background", classes_count[0],
                                count_train_tb)
                tb.add_scalar("Count_Pix_Train/background", pix_classes_count[0],
                                count_train_tb)
                tb.add_scalar("Count_Train/" + str(classes_labels[0]), classes_count[1],
                                count_train_tb)
                tb.add_scalar("Count_Pix_Train/" + str(classes_labels[0]), pix_classes_count[1],
                                count_train_tb)
            else:
                tresholded = torch.argmax(target_var[batch, :, :, :].squeeze(), dim=0)
                tresholded = tresholded.byte()
                classes = torch.unique(tresholded.flatten())
                classes_count[classes] += 1
                for klasa in classes:
                    dummy = torch.zeros([tresholded.shape[0], tresholded.shape[1]])
                    dummy[tresholded == klasa] = 1
                    pix_classes_count[klasa] += torch.sum(dummy)
                    # pix_classes_count[klasa] += np.sum(target_var[batch,klasa, :, :].squeeze().detach().cpu().numpy())

                    tb.add_scalar("Count_Train/" + str(classes_labels[klasa]), classes_count[klasa],
                                    count_train_tb)
                    tb.add_scalar("Count_Pix_Train/" + str(classes_labels[klasa]), pix_classes_count[klasa],
                                    count_train_tb)
        tb.add_scalar("Loss Train Iteration", loss, count_train_tb)

        count_train_tb += 1
    
def tb_write_image(tb, num_classes, epoch, input_var, target_var, model_output, index, train_part,
                   tb_img_name,device,use_mask,dataset,loss_type):
    sigmoid_func = torch.nn.Sigmoid()
    # if num_classes >= 1:
    if loss_type == 'bce':
        model_output = sigmoid_func(model_output)
        tresholded = model_output[index, :, :, :]>0.5
        out = tresholded.byte()
        out = decode_segmap2(out,num_classes,device,loss_type)
        out = torch.moveaxis(out, 2, 0).detach().cpu().numpy()

        target = target_var[index, :, :, :]>0.5
        target = target.byte()
        target = decode_segmap2(target,num_classes,device,loss_type)
        target = torch.moveaxis(target, 2, 0).detach().cpu().numpy()

    elif loss_type == 'ce':
        tresholded = model_output[index, :, :, :]
        out = tresholded.byte()
        out = torch.argmax(out.squeeze(),dim=0)

        out = decode_segmap2(out,num_classes,device,loss_type)
        out = torch.moveaxis(out, 2, 0).detach().cpu().numpy()

        target = target_var[index, :, :, :]
        target = target.byte()
        target = torch.argmax(target.squeeze(),dim=0)
        target = decode_segmap2(target,num_classes,device,loss_type) 
        target = torch.moveaxis(target, 2, 0).detach().cpu().numpy()

    else:
        print("Error: Unimplemented loss type! Importing images to tensorboard interupted")
        sys.exit(0)
    
    image = (input_var[index, :, :, :]).reshape(5, 512, 512)
    #rgb_image = inv_zscore_func(image,dataset)[0:3,:,:]
    #ir_image = inv_zscore_func(image,dataset)[3,:,:]
    rgb_image = (inv_norm_func(image.detach().cpu())[0:3,:,:]).astype('uint8')
    nir_image = inv_norm_func(image.detach().cpu())[3,:,:]
    nir_image = nir_image[np.newaxis,:,:]
    nir_image =  (np.repeat(nir_image,3,axis=0)).astype('uint8')
    red_edge_image = inv_norm_func(image.detach().cpu())[4,:,:]
    red_edge_image = red_edge_image[np.newaxis,:,:]
    red_edge_image =  (np.repeat(red_edge_image,3,axis=0)).astype('uint8') 
    
    
    tb.add_image("RGB , NIR, Red Edge, Label, Prediction" + tb_img_name + " " + train_part,
                    np.concatenate([rgb_image, np.ones(shape=(3,512,10),dtype=np.uint8)*255 , nir_image, 
                    np.ones(shape=(3,512,10),dtype=np.uint8)*255,red_edge_image,np.ones(shape=(3,512,10),dtype=np.uint8)*255,target.astype('uint8'), np.ones(shape=(3,512,10),dtype=np.uint8)*255 ,out.astype('uint8')], axis=2),
                    epoch, dataformats="CHW")
    
    iou_1 = calc_metrics_tb(model_output[index, :, :, :].unsqueeze(dim=0),
                                        target_var[index, :, :, :].unsqueeze(dim=0), num_classes,use_mask)
                                        
    
    iou_1 = torch.tensor(iou_1,dtype = torch.float32)
    tb.add_scalar("Miou/" + tb_img_name + " " + train_part, iou_1, epoch)


def tb_image_list_plotting(tb,tb_img_list,num_channels_lab,epoch,input_var,target_var,model_output,train_part,device,batch_names,use_mask,dataset,loss_type):
    tb_list = [tb_img for tb_img in tb_img_list if tb_img in batch_names]
    if tb_list:
        for tb_list_index in range(len(tb_list)):
            tb_img_index = batch_names.index(tb_list[tb_list_index])
            tb_write_image(tb, num_channels_lab, epoch, input_var, target_var, model_output,
                            tb_img_index, train_part, tb_list[tb_list_index],device,use_mask,dataset,loss_type)
        
        del tb_list, tb_img_index

def tb_add_epoch_losses(tb,train_losses,validation_losses,epoch):
    tb.add_scalar("Loss/Train", torch.mean(torch.tensor(train_losses,dtype = torch.float32)), epoch)
    tb.add_scalar("Loss/Validation", torch.mean(torch.tensor(validation_losses,dtype = torch.float32)), epoch)

def tb_top_k_worst_k(df, num_classes, k_index, test_loader, loss_type, zscore,device,segmentation_net,tb,classes_labels,dataset):
    for class_iter in range(num_classes):
        df_tmp = df[class_iter]
        # klasa0 = df[df['klasa']==i].reset_index().iloc[:,1:]
        mean_num_pix = df_tmp['broj piksela pozitivne klase'].mean()
        std_num_pix = df_tmp['broj piksela pozitivne klase'].std()
        # print("Donji prag broja piksela za pozitivnu klasu: "+ str(mean_num_pix-std_num_pix))
        # print("Gornji prag broja piksela za pozitivnu klasu: "+ str(mean_num_pix+std_num_pix))
        # df_tmp = df_tmp[(df_tmp["broj piksela pozitivne klase"]>(mean_num_pix-std_num_pix)).values & (df_tmp["broj piksela pozitivne klase"]<(mean_num_pix+std_num_pix)).values]
        df_tmp_top2 = df_tmp.reset_index().iloc[:k_index,1:]
        df_tmp_worst2 = df_tmp.reset_index().iloc[-k_index:,1:]
        if not df_tmp_top2.empty and not df_tmp_worst2.empty:
            for k_iter in range(k_index):
                test_img_top_tmp, target_top = load_raw_data(test_loader,df_tmp_top2,k_iter,loss_type)
                test_img_worst_tmp, target_worst = load_raw_data(test_loader,df_tmp_worst2,k_iter,loss_type)

                if zscore:
                    test_img_top = zscore_func(test_img_top_tmp,device,dataset)
                    test_img_worst = zscore_func(test_img_worst_tmp,device,dataset)
                else:
                    test_img_top = norm_func(torch.tensor(test_img_top_tmp),device).permute(2,0,1)
                    test_img_worst = norm_func(torch.tensor(test_img_worst_tmp),device).permute(2,0,1)

                target_top = torch.tensor(target_top)
                target_worst = torch.tensor(target_worst)
                
                
                
                if zscore:
                    image_top = torch.tensor(test_img_top_tmp[:4],device=device)
                    image_worst = torch.tensor(test_img_worst_tmp[:4],device=device)

                    nir_top = inv_zscore_func(image_top,dataset)[3,:,:]
                    rgb_image_top = inv_zscore_func(image_top,dataset)[0:3,:,:]
                    nir_worst = inv_zscore_func(image_worst,dataset)[3,:,:]
                    rgb_image_worst = inv_zscore_func(image_worst,dataset)[0:3,:,:]
                else:
                    image_top = torch.tensor(test_img_top_tmp,device=device).permute(2,0,1)
                    image_worst = torch.tensor(test_img_worst_tmp,device=device).permute(2,0,1)

                    rgb_image_top = inv_norm_func(image_top.detach().cpu())[0:3,:,:]
                    nir_top = inv_norm_func(image_top.detach().cpu())[3,:,:]
                    red_edge_top = inv_norm_func(image_top.detach().cpu())[4,:,:]
                    
                    rgb_image_worst = inv_norm_func(image_worst.detach().cpu())[0:3,:,:]
                    nir_worst = inv_norm_func(image_worst.detach().cpu())[3,:,:]
                    red_edge_worst = inv_norm_func(image_worst.detach().cpu())[4,:,:]
                # nir_top = nir_top.repeat(3,1,1)
                nir_top = nir_top[np.newaxis,:,:]
                nir_top =  np.repeat(nir_top,3,axis=0)
                
                red_edge_top = red_edge_top[np.newaxis,:,:]
                red_edge_top =  np.repeat(red_edge_top,3,axis=0)

                # nir_worst = nir_worst.repeat(3,1,1)
                nir_worst = nir_worst[np.newaxis,:,:]
                nir_worst =  np.repeat(nir_worst,3,axis=0)

                red_edge_worst = red_edge_worst[np.newaxis,:,:]
                red_edge_worst =  np.repeat(red_edge_worst,3,axis=0)
                ## ADD SIGMOID
                sigmoid_func = torch.nn.Sigmoid()
                model_output_top = segmentation_net(test_img_top.unsqueeze(0))
                model_output_top = sigmoid_func(model_output_top)

                out_top = model_output_top[0,:,:,:].squeeze()>0.5
                out_top = out_top.byte().unsqueeze(0)
                out_top = decode_segmap2(out_top, num_classes, device,loss_type)
                out_top = torch.moveaxis(out_top, 2, 0)
                out_top = out_top.detach().cpu().numpy().astype("uint8")

                model_output_worst = segmentation_net(test_img_worst.unsqueeze(0))
                model_output_top = sigmoid_func(model_output_top)
                out_worst = model_output_worst[0,:,:,:].squeeze()>0.5
                out_worst = out_worst.byte().unsqueeze(0)
                out_worst = decode_segmap2(out_worst, num_classes, device,loss_type)
                out_worst = torch.moveaxis(out_worst, 2, 0)
                out_worst = out_worst.detach().cpu().numpy().astype("uint8")

                target_top = decode_segmap2(target_top,num_classes, device,loss_type)
                target_top = torch.moveaxis(target_top,2,0)
                target_top = target_top.detach().cpu().numpy().astype("uint8")

                target_worst = decode_segmap2(target_worst, num_classes, device,loss_type)
                target_worst = torch.moveaxis(target_worst,2,0)
                target_worst = target_worst.detach().cpu().numpy().astype("uint8")


                # tb.add_image("Top 2 Test Images/Classwise_"+ classes_labels[class_iter] + "_top_"+str(k_iter+1)+"_"+df_tmp_top2.iloc[k_iter]['filenames']+ " Class area: "+ str(df_tmp_top2.iloc[k_iter]['broj piksela pozitivne klase']) + " IoU metric: " + str(df_tmp_top2.iloc[k_iter]['iou metrika']) ,
                #                 torch.concat([rgb_image_top.byte(),torch.ones(size=(3,512,10),device=device,dtype=torch.uint8)*255, nir_top.byte() , torch.ones(size=(3,512,10),device=device,dtype=torch.uint8)*255, target_top.byte(), torch.ones(size=(3,512,10),device=device,dtype=torch.uint8)*255, out_top.byte()], axis=2),
                #                 1, dataformats="CHW")
                tb.add_image("Top 2 Test Images/Classwise_" + classes_labels[class_iter] + "_top_" + str(k_iter + 1) + "_" +
                             df_tmp_top2.iloc[k_iter]['filenames'] + " Class area: " + str(
                    df_tmp_top2.iloc[k_iter]['broj piksela pozitivne klase']) + " IoU metric: " + str(
                    df_tmp_top2.iloc[k_iter]['iou metrika']),
                             np.concatenate([rgb_image_top,
                                           np.ones(shape=(3, 512, 10),  dtype=np.uint8) * 255,
                                           nir_top.astype("uint8"),
                                           np.ones(shape=(3, 512, 10),  dtype=np.uint8) * 255,
                                           red_edge_top.astype("uint8"),
                                           np.ones(shape=(3, 512, 10),  dtype=np.uint8) * 255,
                                           target_top,
                                           np.ones(shape=(3, 512, 10), dtype=np.uint8) * 255,
                                           out_top], axis=2),
                             1, dataformats="CHW")
                
                
                # tb.add_image("Worst 2 Test Images/Classwise_"+ classes_labels[class_iter] + "_worst_"+str(k_iter+1)+"_"+df_tmp_worst2.iloc[k_iter]['filenames'] + " Class area: "+ str(df_tmp_top2.iloc[k_iter]['broj piksela pozitivne klase']) +" IoU metric: " + str(df_tmp_worst2.iloc[k_iter]['iou metrika']) ,
                #                 torch.concat([rgb_image_worst.byte(), torch.ones(size=(3,512,10),device=device,dtype=torch.uint8)*255, nir_worst.byte() ,torch.ones(size=(3,512,10),device=device,dtype=torch.uint8)*255, target_worst.byte(),torch.ones(size=(3,512,10),device=device,dtype=torch.uint8)*255, out_worst.byte()], axis=2),
                #                 1, dataformats="CHW")
                tb.add_image(
                    "Worst 2 Test Images/Classwise_" + classes_labels[class_iter] + "_worst_" + str(k_iter + 1) + "_" +
                    df_tmp_worst2.iloc[k_iter]['filenames'] + " Class area: " + str(
                        df_tmp_top2.iloc[k_iter]['broj piksela pozitivne klase']) + " IoU metric: " + str(
                        df_tmp_worst2.iloc[k_iter]['iou metrika']),
                    np.concatenate(
                        [rgb_image_worst, np.ones(shape=(3, 512, 10), dtype=np.uint8) * 255,
                         nir_worst.astype("uint8"), np.ones(shape=(3, 512, 10), dtype=np.uint8) * 255,red_edge_worst.astype("uint8"), np.ones(shape=(3, 512, 10), dtype=np.uint8) * 255,
                         target_worst, np.ones(shape=(3, 512, 10), dtype=np.uint8) * 255,
                         out_worst], axis=2),
                    1, dataformats="CHW")

def createConfusionMatrix(loader,net,classes_labels,loss_type):
    y_pred = [] # save predction
    y_true = [] # save ground truth
    sigmoid_func = torch.nn.Sigmoid()
    for input_var, target_var, img_names_test in loader:
        for idx in range(target_var.shape[0]):
            # if loss_type == "bce": # UNDER CONSTRUCTION
            #     target_tmp = target_var[ idx,:, :, :]>0
            #     target_tmp = target_tmp.byte()
            #     counter = 1
            #     for ch in range(target_tmp.shape[0]):
            #         target_tmp[ch,:,:] = target_tmp[ch,:,:]*counter
            #         counter+=1 
            #     target_tmp = target_tmp.flatten()
            # #   target = target.byte()
            #     # target_conf = torch.argmax(target_var[idx, :, :, :].squeeze(), dim=0).detach().cpu().numpy().flatten()
            #     y_true.extend(target_tmp.detach().cpu().numpy())
            
            if loss_type =="ce" or loss_type =="bce":
                if loss_type == "bce":
                    target_var_bg = (target_var[idx]>0.5).any(0)==False
                    target_var2 = torch.cat([target_var_bg.unsqueeze(0),target_var[idx]],dim = 0)
                else:
                    target_var2 = target_var[idx,:,:]
                # target_conf = torch.argmax(target_var[idx, :, :, :].squeeze(), dim=0).byte().flatten()
                # target_conf2 = (torch.argmax(target_var2.squeeze(), dim=0).byte())[(mask_test[idx,1,:,:]*mask_test[idx,0,:,:]).byte()]
                target_conf2 = (torch.argmax(target_var2.squeeze(), dim=0).byte())
                y_true.extend(target_conf2.unsqueeze(0).detach().cpu().numpy().flatten())
            else:
                print("Error: unimplemented loss type")
                sys.exit(0)
        
        model_output = net(input_var)
        if loss_type == 'bce':
            model_output = sigmoid_func(model_output)
        for idx in range(model_output.shape[0]):
            # if loss_type == "bce": # ALSO UNDER CONSTRUCTION
            #     output_tmp = model_output[ idx,:, :, :]>0
            #     output_tmp = output_tmp.byte()
            #     counter = 1
            #     for ch in range(output_tmp.shape[0]):
            #         output_tmp[ch,:,:] = output_tmp[ch,:,:]*counter
            #         counter+=1 
            #     output_tmp = output_tmp.flatten()
            #     # pred_conf = torch.argmax(model_output[idx, :, :, :].squeeze(), dim=0).detach().cpu().numpy().flatten()
            #     y_pred.extend(output_tmp.detach().cpu().numpy())
            if loss_type =="ce" or loss_type =="bce":
                if loss_type == "bce":
                    #### dodaj sigmoid da model_output bude verovatnoca #####
                    model_output_bg = (model_output[idx]>0.5).any(0)==False
                    model_output2 = torch.cat([model_output_bg.unsqueeze(0),model_output[idx]],dim = 0)
                else:
                    model_output2 = model_output[idx,:,:]
                # pred_conf = torch.argmax(model_output[idx, :, :, :].squeeze(), dim=0).detach().cpu().numpy().flatten()
                # pred_conf2 = (torch.argmax(model_output2.squeeze(), dim=0).byte())[(mask_test[idx,1,:,:]*mask_test[idx,0,:,:]).byte()]
                pred_conf2 = (torch.argmax(model_output2.squeeze(), dim=0).byte())
                y_pred.extend(pred_conf2.unsqueeze(0).detach().cpu().numpy().flatten())

            else:
                print("Error: unimplemented loss type")
                sys.exit(0)
        # pred_conf = np.moveaxis(pred_conf, 1, 3)
        # pred_conf = np.moveaxis(pred_conf, 1, 2)
        # conf_matrix = confusion_matrix(target_conf, pred_conf)

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)


    # conf = np.insert(cf_matrix, cf_matrix.shape[0], np.zeros(len(classes_labels)), axis=1)
    # conf = np.insert(conf, cf_matrix.shape[0], np.zeros(len(classes_labels) + 1), axis=0)
    # for i in range(len(classes_labels)):
    #     conf[i,-1] = np.sum(conf[i,:])
    #     conf[-1, i] = np.sum(conf[:, i])

    # classes = classes_labels
    # classes.append("sum")
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes_labels],
                         columns=[i for i in classes_labels])
    # Create Heatmap
    plt.figure(figsize=(12, 7))
    return sns.heatmap(df_cm, annot=True,xticklabels=True,yticklabels=True).get_figure()