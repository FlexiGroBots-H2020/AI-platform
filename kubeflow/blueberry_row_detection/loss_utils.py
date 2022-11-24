import sys
import numpy as np
import torch.nn as nn
import torch

def loss_init(use_weights,loss_type,dataset,num_channels_lab,device,use_mask):
    if use_weights:
        
        if dataset == "Proba":
            
            if num_channels_lab == 2:
                class_weights = np.load(r'')
            elif num_channels_lab == 1:
                class_weights = np.load(r'')
                # class_weights = [2.19672858]
            else:
                print("Error: wrong dataset")
                sys.exit(0)
        elif dataset == "Borovnice":
    
            if num_channels_lab == 2:
                class_weights = np.load(r'')
            elif num_channels_lab == 1:
                class_weights = np.load(r'')[1:]
            else:
                print("Error: wrong dataset")
                sys.exit(0)
        else:
            print("Error: wrong dataset")
            sys.exit(0)
        
    if loss_type == "ce":
        if use_weights:
            class_weights = torch.tensor(class_weights, dtype=torch.float,device=device)
            criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
        else:
            if use_mask:
                criterion = nn.CrossEntropyLoss(reduction="none")
            else:
                criterion = nn.CrossEntropyLoss(reduction="mean")
        return criterion
    elif loss_type == "bce":
        if use_weights:
            class_weights = torch.tensor(class_weights, dtype=torch.float,device=device).reshape(1, num_channels_lab, 1, 1)
            criterion_bce = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights,reduction="none")
        else:
            if use_mask:
                criterion_bce = torch.nn.BCEWithLogitsLoss(reduction="none")
            else:
                criterion_bce = torch.nn.BCEWithLogitsLoss(reduction="mean")
        return criterion_bce
    elif loss_type == "ce_1":
        if use_weights:
            class_weights = torch.tensor(class_weights, dtype=torch.float,device=device)
            criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
        else:
            if use_mask:
                criterion = nn.CrossEntropyLoss(reduction="none")
            else:
                criterion = nn.CrossEntropyLoss(reduction="mean")
        return criterion

    


def loss_calc(loss_type ,criterion ,model_output ,target_var ,num_channels_lab = 2, use_mask = True): ### num_channels_lab = 2 u slucaju kada imamo 2 klase, bg i fg, Za Saletov slucaj to ce biti 7
                                                                                 ### Kada se koristi bce ili ce kod kog nemamo racunanje verovatnoca argument num_channels_lab nije potrebno    
    if loss_type == "bce":                                                       ### proslediti
        loss = criterion(model_output, target_var)
        if use_mask:
            loss = loss[mask_train.unsqueeze(1).repeat(1,num_channels_lab,1,1)]
            loss = loss.mean()
            return loss
        else:
            return loss

    elif loss_type == 'ce':
        loss = criterion(model_output, torch.argmax(target_var, 1))
        if use_mask:

            loss = loss[mask_train]
        # if use_mask:
        #     loss = torch.multiply(loss, mask_train[:, 0, :, :])
        #     loss = torch.multiply(loss, mask_train[:, 1, :, :])
            loss = loss.mean()
            return loss
        else:
            return loss
    elif loss_type == 'ce_1':
       
        target_var_ce = torch.div(target_var, torch.repeat_interleave(\
            torch.square(torch.sum(target_var, dim=1)).unsqueeze(dim=1), repeats=num_channels_lab, dim=1))
        loss = criterion(model_output, target_var_ce)
        if use_mask:
            loss = torch.multiply(loss, mask_train[:, 0, :, :])
            loss = torch.multiply(loss, mask_train[:, 1, :, :])
            loss = loss.mean()

            return loss
        else:
            return loss