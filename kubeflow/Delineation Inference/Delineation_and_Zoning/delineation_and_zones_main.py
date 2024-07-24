import os
from datetime import datetime
from zoniranje import zone_estimation
import numpy as np
import argparse
from pathlib import Path

def main(number_of_zones, window_size, in_shape, in_raster_red, in_raster_green, in_raster_blue, in_raster_nir, in_raster_rededge, output_dir):
    # window_size = window_size
    # number_of_zones = number_of_zones
    # in_shape = in_shape
    # in_raster_red = in_raster_red
    # in_raster_green = in_raster_green
    # in_raster_blue = in_raster_blue
    # in_raster_nir = in_raster_nir
    # in_raster_rededge = in_raster_rededge
    # out_shp = out_shp
    # number_of_zones = 4
    # window_size = 10.0 # in meters !!! # parametar za ukrupnjavanje

    #Paths to tif and shp files
    # in_shape = "C:\\Users\\daniv\\Downloads\\TestParcel_Belanovica1.shp"
    # in_raster_red = "C:\\Users\\daniv\\Downloads\\Belanovica1_processed_transparent_mosaic_red.tif"
    # in_raster_green = "C:\\Users\\daniv\\Downloads\\Belanovica1_processed_transparent_mosaic_green.tif"
    # in_raster_blue = "C:\\Users\\daniv\\Downloads\\Belanovica1_processed_transparent_mosaic_blue.tif"
    # in_raster_nir = "C:\\Users\\daniv\\Downloads\\Belanovica1_processed_transparent_mosaic_nir.tif"
    # in_raster_rededge = "C:\\Users\\daniv\\Downloads\\Belanovica1_processed_transparent_mosaic_red edge.tif"
    #Path to output shp file
    
    current_datetime = datetime.now()
    current_datetime_str = current_datetime.strftime("%H_%M-%d-%m-%Y")
    folder_name = output_dir
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    out_shp = folder_name + "output_sampling_points.shp"
    zone_estimation(in_raster_red,
                    in_raster_green,
                    in_raster_blue,
                    in_raster_nir,
                    in_raster_rededge,
                    in_shape, 
                    folder_name,
                    out_shp, 
                    window_size, 
                    number_of_zones)


if __name__ == "__main__":
    print("Starting Delineation Component")
    print("Parsiranje Input Argumenata")
    parser = argparse.ArgumentParser(description='My program description')
    parser.add_argument('--number_of_zones', type=int, default=4)
    parser.add_argument('--window_size', type=float, default=10)
    parser.add_argument('--in_shape', type=str)
    parser.add_argument('--in_raster_red', type=str)
    parser.add_argument('--in_raster_green', type=str)
    parser.add_argument('--in_raster_blue', type=str)
    parser.add_argument('--in_raster_nir', type=str)
    parser.add_argument('--in_raster_rededge', type=str)
    parser.add_argument('--path_to_save', type=str)
    parser.add_argument('--new_location', type=str)

    print("Kraj Input Argova")
    parser.add_argument('--output1-path', type=str, help='Path of the local file where the Output 1 data should be written.')
    print('Kraj Output Argova')
    args = parser.parse_args()
    print('Kraj Parsiranja')
    print("Pokretanje Maina")
    in_shape = args.in_shape
    in_raster_red = args.in_raster_red
    in_raster_green = args.in_raster_green
    in_raster_blue = args.in_raster_blue
    in_raster_nir = args.in_raster_nir
    in_raster_rededge = args.in_raster_rededge
    number_of_zones = args.number_of_zones
    window_size = args.window_size
    path_to_save = args.path_to_save
    new_loc = args.new_location
    print(number_of_zones)
    print(window_size)
    print(in_shape)
    print(in_raster_red)
    print(in_raster_green)
    print(in_raster_blue)
    print(in_raster_nir)
    print(in_raster_rededge)
    print(path_to_save)
    print(new_loc)
    main(number_of_zones, 
         window_size, 
         in_shape, 
         in_raster_red, 
         in_raster_green, 
         in_raster_blue, 
         in_raster_nir, 
         in_raster_rededge, 
         path_to_save)
    print("End of Component")
    Path(args.output1_path).parent.mkdir(parents=True, exist_ok=True)
    print('Creating a emptydir to link components')
    with open(args.output1_path, 'w') as output1_file:
        _ = output1_file.write("empty_cache")