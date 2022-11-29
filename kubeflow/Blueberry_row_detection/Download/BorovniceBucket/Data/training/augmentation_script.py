import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

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
            