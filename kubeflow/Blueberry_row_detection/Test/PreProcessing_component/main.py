
import numpy as np

import copy



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

def main(color_load, border_shp, save_test_path, input_files_type=None):



    if input_files_type == "npy":
        in_test_raster_r = color_load + "test_ch_red_croped.npy"
        in_test_raster_g = color_load + "test_ch_green_croped.npy"
        in_test_raster_b = color_load + "test_ch_blue_croped.npy"
        in_test_raster_rededge = color_load + "test_ch_red_edge_croped.npy"
        in_test_raster_nir = color_load + "test_ch_nir_croped.npy"
        border_shp = border_shp+ "test_parcela_shape.shp"
        ch_red = np.load(in_test_raster_r)
        ch_green = np.load(in_test_raster_g)
        ch_blue = np.load(in_test_raster_b)
        ch_nir = np.load(in_test_raster_nir)
        ch_rededge = np.load(in_test_raster_rededge)

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
    save_test_data_pth = save_test_path

    stacked_geotiffs_npy = np.stack([ch_red,ch_green,ch_blue,ch_rededge,ch_nir,cropped_mask],axis = 2)
    preprocessing_block(stacked_geotiffs_npy, cropped_mask, save_test_data_pth)


if __name__ == '__main__':
    color_load = "/home/tloken/biosens/borovnice/Data/kanali_npy/"
    border_shp = "/home/tloken/biosens/borovnice/Data/shp/"
    save_test_path = "/home/tloken/biosens/borovnice/Data/test_data_folder"

    print("==========================START PREPROCESSING BLOCK==========================")
    main(color_load = color_load, border_shp = border_shp, save_test_path = save_test_path, input_files_type = "npy")

    print("==========================END PREPROCESSING BLOCK==========================")
