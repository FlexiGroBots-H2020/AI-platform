
def test_block(load_data_pth,model_pth,save_pred_pth):
    print("Test block started")
    device = 'cpu'
    segmentation_net = smp.UnetPlusPlus(in_channels=5, encoder_depth=3, classes=1,activation=None,decoder_channels=[64, 32, 16]).to(device=device)
    # segmentation_net = UNet3(n_channels=5, n_classes=1, height=512, width= 512, zscore = False)
    load_checkpoint(torch.load(model_pth,map_location=torch.device('cpu')),segmentation_net)
    # segmentation_net.load_state_dict()
    # load_checkpoint(torch.load(model_pth,map_location=torch.device('cpu'))['state_dict'],segmentation_net)
    # load_checkpoint(torch.load(r'/home/stefanovicd/DeepSleep/Borovnice/UNET_YT_VERSION/my_checkpoint_unetplusplus__39_masked_lr_1e-05.pth.tar',map_location=torch.device(DEVICE))['state_dict'],model)
    # segmentation_net.load_state_dict(torch.load(model_pth,map_location=torch.device('cpu')))
    segmentation_net.eval()

    import re
    samples = os.listdir(load_data_pth)
    samples = sorted(samples, key=lambda s: int(re.search(r'\d+', s).group()))
    for sample in samples:
        img = np.load(load_data_pth + "/" + sample)
        img = torch.tensor(img).permute(2,0,1).unsqueeze(0)
        pred_sample = segmentation_net(img.float())
        sigmoid_func = torch.nn.Sigmoid()
        preds_sample2 = sigmoid_func(pred_sample)
        pred_sample_bin = (preds_sample2 > 0.5).byte()

        # plt.imsave(save_pred_pth+"/pred_"+sample[:-4]+'.png',pred_sample_bin.cpu().detach().numpy()[0][0])
        np.save(save_pred_pth+"/pred_"+sample,pred_sample_bin.cpu().detach().numpy()[0][0])

    # model.load_state_dict(torch.load(r'/home/stefanovicd/DeepSleep/GraphNeuralNetworks/Deeplab-Large-FOV-master/saved_models/GCNModel_2000segmenata_5Sobel_kanala_treningVeci_test_h128_lr2nule_k16.pt',map_location=torch.device('cpu')))
    print("Test block ended")

save_test_data_pth = r"/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/test_data_folder"
load_test_data_pth = copy.deepcopy(save_test_data_pth+"/img")
save_pred_data_pth = r"/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/test_pred_folder"
model_pth = r"/home/stefanovicd/DeepSleep/agrovision/BorovniceUnetBS/logs/Train_BGFG_BCE_with_weights/0_13_11_2022_01_36_47_lr_1e-05_step_na_5_epoha_lambda_parametar_1_batch_size_4_sched_multiplicative_loss_bce/NN_model_ep_40_Train_BGFG_BCE_with_weights/fully_trained_model_epochs_39_lr_1e-05_step_5_Lambda_parametar_1_loss_type_bce_arhitektura_UNet++_batch_size_4.pt"

test_block(load_test_data_pth,model_pth,save_pred_data_pth)
