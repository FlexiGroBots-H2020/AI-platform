from calendar import EPOCH
from signal import pthread_sigmask
from typing import Final
from unittest.mock import patch
import matplotlib.pyplot as plt
from numpy import binary_repr
import torch.utils.data.dataloader
from osgeo import gdal,ogr,osr
import torch
from torch.utils.tensorboard import SummaryWriter
import random
from Unet_LtS import UNet3
import segmentation_models_pytorch as smp
# from torchsummary import summary
import os
from Train_BGFG_BCE_with_weightsUnet3 import main
from print_utils import *
from data_utils import *
from loss_utils import *
from model_utils import *
from tb_utils import *
from metrics_utils import*
from configUnet3 import config_func_unet3
# from focal_loss import FocalLoss2
import time
import numpy as np
import time
import matplotlib.pyplot as plt

import cv2

SHRT_MAX = 32767

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


def augmentation_and_saving(img,label,path_img,path_label,testing_flag = True,png_flag = False):

    if png_flag:
        img = np.array(img,'uint8')

    if len(path_img.split()[-1].split("_")[-1]) == 1:
        index = -18
    elif len(path_img.split()[-1].split("_")[-1]) > 1 and len(path_img.split()[-1].split("_")[-1]) < 3 :
        index = -19
    elif len(path_img.split()[-1].split("_")[-1]) > 2 :
        index = -20
    
    if png_flag:
        plt.imsave(path_label[:index]+'label_' +path_label[index:] + '.png', label)
        plt.imsave(path_img + '.png', img[:,:,:3])
    else:
        np.save(path_label[:index]+'label_' +path_label[index:] + '.npy', label)
        np.save(path_img + '.npy', img)
    if testing_flag == False:
        
        rot1_label = np.rot90(label)
        rot2_label = np.rot90(rot1_label)
        rot3_label = np.rot90(rot2_label)
        rot1_img = np.rot90(img)
        rot2_img = np.rot90(rot1_img)
        rot3_img = np.rot90(rot2_img)
        

        if png_flag:
            plt.imsave(path_label[:index]+'label_' +path_label[index:] + '_rot90' + '.png', rot1_label)
            plt.imsave(path_label[:index]+'label_' +path_label[index:] + '_rot180' + '.png', rot2_label)
            plt.imsave(path_label[:index]+'label_' +path_label[index:] + '_rot270' + '.png', rot3_label)
            plt.imsave(path_img + '_rot90' + '.png', rot1_img[:,:,:3])
            plt.imsave(path_img + '_rot180' + '.png', rot2_img[:,:,:3])
            plt.imsave(path_img + '_rot270' + '.png', rot3_img[:,:,:3])
        else:
            np.save(path_label[:index]+'label_' +path_label[index:] + '_rot90'+'.npy', rot1_label)
            np.save(path_label[:index]+'label_' +path_label[index:] + '_rot180' + '.npy', rot2_label)
            np.save(path_label[:index]+'label_' +path_label[index:] + '_rot270' + '.npy', rot3_label)
            np.save(path_img + '_rot90' + '.npy', rot1_img)
            np.save(path_img + '_rot180' + '.npy', rot2_img)
            np.save(path_img + '_rot270' + '.npy', rot3_img)
        label = cv.flip(label,flipCode=1)
        img = cv.flip(img,flipCode=1)
        
        rot1_label = np.rot90(label)
        rot2_label = np.rot90(rot1_label)
        rot3_label = np.rot90(rot2_label)
        rot1_img = np.rot90(img)
        rot2_img = np.rot90(rot1_img)
        rot3_img = np.rot90(rot2_img)
        
        
        if png_flag:
            plt.imsave(path_label[:index]+'label_' +path_label[index:]+ '_fliped.png', label)
            plt.imsave(path_img + '_fliped.png', img[:, :, :3])
            plt.imsave(path_label[:index]+'label_' +path_label[index:] + '_rot90_fliped' + '.png', rot1_label)
            plt.imsave(path_label[:index]+'label_' +path_label[index:] + '_rot180_fliped' + '.png', rot2_label)
            plt.imsave(path_label[:index]+'label_' +path_label[index:] + '_rot270_fliped' + '.png', rot3_label)
            plt.imsave(path_img + '_rot90_fliped' + '.png', rot1_img[:,:,:3])
            plt.imsave(path_img + '_rot180_fliped' + '.png', rot2_img[:,:,:3])
            plt.imsave(path_img + '_rot270_fliped' + '.png', rot3_img[:,:,:3])
        else:
            np.save(path_label[:index]+'label_' +path_label[index:] + '_fliped.npy', label)
            np.save(path_img + '_fliped.npy', img)
            np.save(path_label[:index]+'label_' +path_label[index:] + '_rot90_fliped' + '.npy', rot1_label)
            np.save(path_label[:index]+'label_' +path_label[index:] + '_rot180_fliped' + '.npy', rot2_label)
            np.save(path_label[:index]+'label_' +path_label[index:] + '_rot270_fliped' + '.npy', rot3_label)
            np.save(path_img + '_rot90_fliped' + '.npy', rot1_img)
            np.save(path_img + '_rot180_fliped' + '.npy', rot2_img)
            np.save(path_img + '_rot270_fliped' + '.npy', rot3_img)
            



def preprocessing_block(GeoTiff,Shape,save_data_pth):
    print("Preprocessing block started")
    print("Dividing test parcel to 512x512 patches")
    
    H = np.shape(GeoTiff[:,:,0])[0]
    W = np.shape(GeoTiff[:,:,0])[1]
    patch_size = 512
    tmp_labels = np.zeros([patch_size,patch_size])
    tmp_rgb = np.zeros([patch_size,patch_size,3])
    tmp_nir = np.zeros([patch_size,patch_size])
    tmp_red_edge = np.zeros([patch_size,patch_size])
    tmp_svi_kanali_combo = np.zeros([patch_size,patch_size])
    
    testing_flag = True
    counter = 0
    for i in range(H//patch_size):
        
        for j in range(W//patch_size):
            
            tmp_labels = Shape[i*patch_size:i*patch_size+patch_size,j*patch_size:j*patch_size+patch_size]
                
            tmp_rgb = GeoTiff[i*patch_size:i*patch_size+patch_size,j*patch_size:j*patch_size+patch_size,:3]
            tmp_nir = GeoTiff[i * patch_size:i * patch_size + patch_size, j * patch_size:j * patch_size + patch_size,3]
            tmp_nir = np.expand_dims(tmp_nir, axis=2)
            tmp_red_edge = GeoTiff[i * patch_size:i * patch_size + patch_size, j * patch_size:j * patch_size + patch_size,4]
            tmp_red_edge = np.expand_dims(tmp_red_edge, axis=2)
            tmp_svi_kanali_combo = np.concatenate([tmp_rgb,tmp_nir,tmp_red_edge],axis=2)
            png_flag = False
            if np.sum(tmp_labels)!= 0:
                
                if png_flag:
                    path_img = save_data_pth + '/img/svi_kanali_combo_' + str(counter)
                    path_label = save_data_pth + '/label/svi_kanali_combo_' + str(counter)
                    augmentation_and_saving(tmp_svi_kanali_combo, tmp_labels, path_img, path_label, testing_flag, png_flag)
                else:
                    path_img = save_data_pth + '/img/svi_kanali_combo_' + str(counter)
                    path_label = save_data_pth + '/label/svi_kanali_combo_' + str(counter)
                    augmentation_and_saving(tmp_svi_kanali_combo, tmp_labels, path_img, path_label, testing_flag, png_flag)

                print('sample '+str(counter))
                counter+=1

    print("Test data saved")
    print("Preprocessing block ended")


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
def postprocessing_block(GeoTiff,y0,y1,x0,x1,final_mask, geotransform, projection, load_pred_pth,save_final_GeoTiff_pth,save_final_colored_GeoTiff_pth,save_final_shp_pth):
    print("Postprocessing block started")

    H = np.shape(GeoTiff[:,:,0])[0]
    W = np.shape(GeoTiff[:,:,0])[1]
    patch_size = 512
    tmp_labels = np.zeros([patch_size,patch_size])
    tmp_rgb = np.zeros([patch_size,patch_size,3])
    # tmp_nir = np.zeros([patch_size,patch_size])
    # tmp_red_edge = np.zeros([patch_size,patch_size])
    tmp_svi_kanali_combo = np.zeros([patch_size,patch_size])
    Final_prediction = np.zeros_like(GeoTiff[:,:,0],dtype='uint8')
    testing_flag = True
    import re
    samples = os.listdir(load_pred_pth)
    samples = sorted(samples, key=lambda s: int(re.search(r'\d+', s).group()))
    counter = 0
    for i in range(H//patch_size):
        
        for j in range(W//patch_size):
            
            tmp_labels = GeoTiff[i*patch_size:i*patch_size+patch_size,j*patch_size:j*patch_size+patch_size,5]
                
            
            png_flag = False
            if np.sum(tmp_labels)!= 0:
                
                Final_prediction[i*patch_size:i*patch_size+patch_size,j*patch_size:j*patch_size+patch_size] = np.load(load_pred_pth+'/'+samples[counter])

                print('Final test sample '+str(counter))
                counter+=1

    # # create the 3-band raster file
    # dst_ds = gdal.GetDriverByName('GTiff').Create('myGeoTIFF.tif', W, H, 1, gdal.GDT_Byte)

    # dst_ds.SetGeoTransform(geotransform)    # specify coords
    # srs = osr.SpatialReference()            # establish encoding
    # srs.ImportFromEPSG(32634)                # WGS84 lat/long
    # dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
    # dst_ds.GetRasterBand(1).WriteArray(Final_prediction)   # write r-band to the raster
    # # dst_ds.GetRasterBand(2).WriteArray(g_pixels)   # write g-band to the raster
    # # dst_ds.GetRasterBand(3).WriteArray(b_pixels)   # write b-band to the raster
    # dst_ds.FlushCache()                     # write to disk

    srsProj = osr.SpatialReference()
    srsProj.ImportFromEPSG(32634)
    final_mask[y0:y1,x0:x1] = copy.deepcopy(Final_prediction)
    # final_mask[y0 - row_pad:y_end + row_pad, x_start - column_pad:x_end + column_pad] = copy.deepcopy(image)
    driver = gdal.GetDriverByName('GTiff')
    tmp_raster = driver.Create('test_parcel.tif', final_mask.shape[1], final_mask.shape[0], 1, gdal.GDT_Byte)
    tmp_raster.SetGeoTransform(geotransform)
    tmp_raster.SetProjection(projection)
    srcband = tmp_raster.GetRasterBand(1)
    srcband.WriteArray(final_mask)
    srcband.FlushCache()
    print("Postprocessing block ended")


def main(input_files_type=None):
    
    # Loading test rasters
    
    if input_files_type == "npy":
        in_test_raster_r = r"/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/kanali npy/test_ch_red_croped.npy"
        in_test_raster_g = r"/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/kanali npy/test_ch_green_croped.npy"
        in_test_raster_b = r"/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/kanali npy/test_ch_blue_croped.npy"
        in_test_raster_rededge = r"/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/kanali npy/test_ch_red_edge_croped.npy"
        in_test_raster_nir = r"/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/kanali npy/test_ch_nir_croped.npy"
        border_shp = r'/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/shp/test_parcela_shape.shp'
        ch_red = np.load(in_test_raster_r)
        ch_green = np.load(in_test_raster_g)
        ch_blue = np.load(in_test_raster_b)
        ch_nir = np.load(in_test_raster_nir)
        ch_rededge = np.load(in_test_raster_rededge)
    
    else:
        in_raster_r = r"/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/GeoTiffs/Babe_registrated_corrected_transparent_mosaic_red.tif"
        in_raster_g = r"/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/GeoTiffs/Babe_registrated_corrected_transparent_mosaic_green.tif"
        in_raster_b = r"/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/GeoTiffs/Babe_registrated_corrected_transparent_mosaic_blue.tif"
        in_raster_rededge = r"/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/GeoTiffs/Babe_registrated_corrected_transparent_mosaic_red edge.tif"
        in_raster_nir = r"/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/GeoTiffs/Babe_registrated_corrected_transparent_mosaic_nir.tif"
        border_shp = r'/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/shp/test_parcela_shape.shp'
        r = gdal.Open(in_raster_r)
        g = gdal.Open(in_raster_g)
        b = gdal.Open(in_raster_b)
        nir = gdal.Open(in_raster_nir)
        rededge = gdal.Open(in_raster_rededge)

        # tmp_geo_transform = r.GetGeoTransform()

        ch_red = r.GetRasterBand(1).ReadAsArray()  # .astype("float64")
        ch_green = g.GetRasterBand(1).ReadAsArray()  # .astype("float64")
        ch_blue = b.GetRasterBand(1).ReadAsArray()  # .astype("float64")
        ch_nir = nir.GetRasterBand(1).ReadAsArray()
        ch_rededge = rededge.GetRasterBand(1).ReadAsArray()

        wkt_raster_proj = r.GetProjectionRef()
        proj_raster = osr.SpatialReference(wkt_raster_proj)
        units = proj_raster.GetAttrValue('UNIT', 0)

        border_shape = ogr.Open(border_shp)
        lyr = border_shape.GetLayer()
        proj_shp = lyr.GetSpatialRef()
        equal_proj = proj_raster.IsSameGeogCS(proj_shp)


        if units == 'degree' or equal_proj != 1:
            raise Exception(
                "Raster and shapefile don't have the same projection or pixel resolution is not in meters. Input data have to satisfy both conditions.")
        cols = r.RasterXSize
        rows = r.RasterYSize
        final_mask = np.zeros(shape=(rows, cols), dtype='uint8')
        # final_pts = np.zeros([len(lyr),5,2])
        final_pts = []
        # pocetak for petlje za masku
        for i in range(len(lyr)):
            feat = lyr.GetFeature(i)
            geom = feat.GetGeometryRef()

            # another check of image size !!! with algoritham for image croping !!!
            # read transform from raster
            geo_transform = r.GetGeoTransform()
            xOrigin = geo_transform[0]
            yOrigin = geo_transform[3]
            pixelWidth = geo_transform[1]
            pixelHeight = geo_transform[5]

            pts_geo = geom.GetGeometryRef(0)
            pts = np.zeros(shape=(pts_geo.GetPointCount(), 2), dtype='int32')
            count = 0

            for p in range(pts_geo.GetPointCount()):
                pts[count, 0] = int((pts_geo.GetX(p) - xOrigin) / pixelWidth)
                pts[count, 1] = int((pts_geo.GetY(p) - yOrigin) / pixelHeight)
                count += 1
            # final_pts[i,:,:]=pts
            if i == 0:
                final_pts.append(pts)
            else:
                # final_pts = [final_pts,pts]
                # final_pts = np.concatenate()
                final_pts.append(pts)
            mask = np.zeros(shape=(rows, cols), dtype='uint8')
            cv2.fillPoly(mask, [pts], 255)
            mask = (mask[:, :] == 255)
            final_mask = final_mask + mask
            print(i)

        final_final_pts = []
        for i in range(len(final_pts)):
            if i ==0:
                final_final_pts = np.concatenate([np.reshape(final_final_pts,[0,2]),final_pts[i]])
            else:
                final_final_pts = np.concatenate([final_final_pts, final_pts[i]])
        
        final_mask[final_mask==2] = 0
    
        x0, y0 = np.min(final_final_pts, axis=0)
        x1, y1 = np.max(final_final_pts, axis=0) + 1  # slices are exclusive at the top
        x0 = np.array(x0, 'int')
        y0 = np.array(y0, 'int')
        x1 = np.array(x1, 'int')
        y1 = np.array(y1, 'int')

        # Get the contents of the bounding box.
        cropped_mask = copy.deepcopy(final_mask[y0:y1, x0:x1])
        diag_len = int(np.sqrt(cropped_mask.shape[0]**2 + cropped_mask.shape[1]**2))

        if (SHRT_MAX < diag_len):
            raise Exception ("Image size exceeds allow size !!!")
        
        ch_red = ch_red[y0:y1,x0:x1]*cropped_mask
        ch_green = ch_green[y0:y1,x0:x1]*cropped_mask
        ch_blue = ch_blue[y0:y1,x0:x1]*cropped_mask
        ch_rededge = ch_rededge[y0:y1,x0:x1]*cropped_mask
        ch_nir = ch_nir[y0:y1,x0:x1]*cropped_mask

    ch_red = (ch_red - np.min(ch_red)) / (np.max(ch_red) - np.min(ch_red))*255
    ch_green = (ch_green - np.min(ch_green)) / (np.max(ch_green) - np.min(ch_green))*255
    ch_blue = (ch_blue - np.min(ch_blue)) / (np.max(ch_blue) - np.min(ch_blue))*255
    ch_nir = (ch_nir - np.min(ch_nir)) / (np.max(ch_nir) - np.min(ch_nir))*255
    ch_rededge = (ch_rededge - np.min(ch_rededge)) / (np.max(ch_rededge) - np.min(ch_rededge))*255
    stacked_geotiffs_npy = np.stack([ch_red,ch_green,ch_blue,ch_rededge,ch_nir,cropped_mask],axis = 2)
    save_test_data_pth = r"/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/test_data_folder"
    # border_shp = r'/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/shp/test_parcela_shape.shp'
    preprocessing_block(stacked_geotiffs_npy, cropped_mask, save_test_data_pth)
    
    
    load_test_data_pth = copy.deepcopy(save_test_data_pth+"/img")
    save_pred_data_pth = r"/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/test_pred_folder"
    model_pth = r"/home/stefanovicd/DeepSleep/agrovision/BorovniceUnetBS/logs/Train_BGFG_BCE_with_weights/0_13_11_2022_01_36_47_lr_1e-05_step_na_5_epoha_lambda_parametar_1_batch_size_4_sched_multiplicative_loss_bce/NN_model_ep_40_Train_BGFG_BCE_with_weights/fully_trained_model_epochs_39_lr_1e-05_step_5_Lambda_parametar_1_loss_type_bce_arhitektura_UNet++_batch_size_4.pt" 
    
    test_block(load_test_data_pth,model_pth,save_pred_data_pth)
    
    load_pred_data_pth = copy.deepcopy(save_pred_data_pth)
    
    save_final_GeoTiff_pth = r'/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/final_results_folder'
    save_final_colored_GeoTiff_pth = r'/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/final_results_folder'
    save_final_shp_pth = r'/home/stefanovicd/DeepSleep/agrovision/DetekcijaBorovnica/final_results_folder'
    postprocessing_block(stacked_geotiffs_npy,y0,y1,x0,x1,final_mask, geo_transform,wkt_raster_proj,load_pred_data_pth, save_final_GeoTiff_pth, save_final_colored_GeoTiff_pth, save_final_shp_pth)



if __name__ == '__main__':
    main()
    


