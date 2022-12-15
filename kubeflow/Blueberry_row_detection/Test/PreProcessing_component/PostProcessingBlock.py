from osgeo import gdal,ogr,osr


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
