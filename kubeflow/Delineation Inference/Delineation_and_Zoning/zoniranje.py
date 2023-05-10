import time
import numpy as np
import os
from utils_calc_zones import create_mask_from_shp_v2
from utils_calc_zones import index_calculation_v2, vegetation_detection_v2, clusterization_of_vegetation_coverage_v2
from utils_calc_zones import generate_decision_zones_map_with_sampling_points, sampling_points_generator
from utils_calc_zones import statistics_from_map, create_shp_multipolygon_from_raster

def zone_estimation(in_raster_red,
                    in_raster_green,
                    in_raster_blue, 
                    in_raster_nir, 
                    in_raster_rededge, 
                    in_shape,  
                    folder_name,
                    out_shp, 
                    window_size = 10, 
                    number_of_zones = 2):

    try:
        flag_info = 0
        start = time.time()

        (B_ch, G_ch, R_ch, NIR_ch, RedEdge_ch, mask_ROI, pixelSizeX, 
         pixelSizeY, in_raster_red, in_raster_green, in_raster_blue, 
         in_raster_nir, in_raster_rededge, y_start, x_start, y_end, x_end) = create_mask_from_shp_v2(in_raster_red, 
                                                                                                     in_raster_green, 
                                                                                                     in_raster_blue, 
                                                                                                     in_raster_nir, 
                                                                                                     in_raster_rededge, 
                                                                                                     in_shape)
        mask_ROI_tmp = mask_ROI.copy()
        end = time.time()
        print('Image croping and reading of geotiff image channels ', end - start)
    except Exception as e:
        flag_info = 0
        if e.args[0] == "Raster and shapefile don't have the same projection or pixel resolution is not in meters. Input data have to satisfy both conditions.":
            flag_info = 1
        if e.args[0] == "Image size exceeds allow size !!!":
            flag_info = 2
        elif e.args[0] == "Shapefile exceeds parcel region !!!":
            flag_info = 3

        print(e)
        return flag_info
    else:
        # calculation of indices 28 seconds
        start = time.time()
        ExG, CIVE, TGI, GLI, NDVI, WDRVI, MGRV, MPRI, RGBVI = index_calculation_v2(R_ch, G_ch, B_ch,NIR_ch ,RedEdge_ch, mask_ROI)
        end = time.time()
        print('Calculation of indices ', end - start)

        # segmentation of vegetation
        start = time.time()
        V_mask = vegetation_detection_v2(WDRVI, MGRV, mask_ROI)
        end = time.time()
        print('Segmentation of vegetation ', end - start)

        # clusterization of vegetation cover
        start = time.time()
        cluster_img = clusterization_of_vegetation_coverage_v2(ExG, NDVI, WDRVI, MGRV, MPRI, RGBVI, V_mask, number_of_zones)
        end = time.time()
        print('Clusterization of vegetation ', end - start)

        pixel_size = min(pixelSizeX, np.abs(pixelSizeY))
        window_size = int(np.ceil(window_size / pixel_size))

        start = time.time()
        cluster_img[:, :], num_zones, map_index, sampling_points = generate_decision_zones_map_with_sampling_points(cluster_img,
                                                                                              mask_ROI,
                                                                                              number_of_zones,
                                                                                              window_size,
                                                                                              y_start,
                                                                                              x_start,
                                                                                              mask_ROI_tmp)
        end = time.time()

        #Writing the previously generated sampling points to the output shp file.
        print('Generate sampling points ', end - start)
        start = time.time()
        sampling_points_generator(in_raster_red, sampling_points, out_shp)
        end = time.time()
        print('Write sampling points in shp ', end - start)

        start = time.time()
        stats_structure = statistics_from_map(cluster_img, num_zones, map_index, V_mask, mask_ROI, ExG, CIVE, TGI, GLI)
        end = time.time()
        print('Calculation of statistic from dec_map ', end - start)
        folder_name = folder_name + "/TEMP/"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        out_tmp_shp = folder_name + "output.shp"
        create_shp_multipolygon_from_raster(in_raster_red, 
                                            num_zones, 
                                            map_index, 
                                            cluster_img, 
                                            out_tmp_shp, 
                                            out_shp, 
                                            stats_structure, 
                                            y_start, 
                                            y_end, 
                                            x_start, 
                                            x_end)
        return flag_info