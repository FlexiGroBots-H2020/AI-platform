from Unet_LtS import UNet3, UNet3_modified
from SegNet import SegResNet
from UperNet import UperNet
from PSPNet import PSPDenseNet
from DUC_HDCNet import DeepLab_DUC_HDC
import segmentation_models_pytorch as smp
import torch

def set_zero_grad(model):
    for param in model.parameters():
        param.grad = None

def model_init(num_channels,num_channels_lab,img_h,img_w,zscore,net_type,device,server,GPU_list):
    if net_type == "UNet3":
        segmentation_net = UNet3(n_channels=num_channels, n_classes=num_channels_lab, height=img_h, width= img_w, zscore = zscore)
    elif net_type == "UNet3_modified":
        segmentation_net = UNet3_modified(num_channels, num_channels_lab, img_h, img_w, 50)
    elif net_type == "UNet++":
        segmentation_net = smp.UnetPlusPlus(in_channels=num_channels, encoder_depth=3, classes=num_channels_lab,
                                                activation=None,decoder_channels=[64, 32, 16]).to(device=device)
    elif net_type == "SegNet":
        segmentation_net = SegResNet(num_classes = num_channels_lab, in_channels = num_channels)
    elif net_type == "PSPNet":
        segmentation_net = PSPDenseNet(num_classes = num_channels_lab, in_channels = num_channels)
    elif net_type == "UperNet":
        segmentation_net = UperNet(num_classes = num_channels_lab, in_channels = num_channels)
    elif net_type == "DUC_HDCNet":
        segmentation_net = DeepLab_DUC_HDC(num_classes = num_channels_lab, in_channels = num_channels)
        
    segmentation_net.to(device)
    if server:
        segmentation_net = torch.nn.DataParallel(segmentation_net, device_ids=GPU_list)

    return segmentation_net

def optimizer_init(segmentation_net,lr,weight_decay,scheduler_lr,lambda_parametri,optimizer_patience):
    if weight_decay != 0:
        optimizer = torch.optim.Adam(params=segmentation_net.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(params=segmentation_net.parameters(), lr=lr)

    if scheduler_lr == 'lambda':
        lmbda = lambda epoch: 0.99
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lmbda])
    elif scheduler_lr == 'multiplicative':
        lmbda = lambda epoch: lambda_parametri
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda, last_epoch=- 1, verbose=False)
    elif scheduler_lr == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    elif scheduler_lr == 'reducelr':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=optimizer_patience)
    elif scheduler_lr == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=lr)
    else:
        scheduler_lr = False

    return optimizer , scheduler
