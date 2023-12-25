import numpy as np
import cmath, copy, cv2, joblib
from osgeo import gdal, osr, ogr
import os
from sklearn.cluster import MiniBatchKMeans
from skimage.filters import threshold_otsu
from sklearn.preprocessing import StandardScaler
from skimage.morphology import skeletonize,dilation,erosion, disk,rectangle
from cython_function import calculation_of_TGI_and_GLI
from cython_function import calculation_of_WDRVI_and_CCCI_and_NDVI_and_CIrededge
from cython_function import calculation_of_ExG_and_CIVE
from cython_function import calculation_of_MGRV_and_MPRI_and_RGBVI
from cython_function import prob_map_calc

colors_grey = np.array([255, 200, 155, 100, 55], dtype="uint8")
SHRT_MAX = 32767

def create_mask_from_shp_v2(grid_raster_filename_red, 
                            grid_raster_filename_green, 
                            grid_raster_filename_blue, 
                            grid_raster_filename_nir, 
                            grid_raster_filename_rededge, 
                            border_shp_filename):

    #open raster file and reproject it if needed
    raster_red = gdal.Open(grid_raster_filename_red)
    raster_green = gdal.Open(grid_raster_filename_green)
    raster_blue = gdal.Open(grid_raster_filename_blue)
    raster_nir = gdal.Open(grid_raster_filename_nir)
    raster_rededge = gdal.Open(grid_raster_filename_rededge)
    #get geospetial info of image
    wkt_raster_proj = raster_red.GetProjectionRef()
    proj_raster = osr.SpatialReference(wkt_raster_proj)
    units = proj_raster.GetAttrValue('UNIT', 0)
    border_shape = ogr.Open(border_shp_filename)
    lyr = border_shape.GetLayer()
    proj_shp = lyr.GetSpatialRef()
    equal_proj = proj_raster.IsSameGeogCS(proj_shp)

    if units == 'degree' or equal_proj != 1:
        raise Exception ("Raster and shapefile don't have the same projection or pixel resolution is not in meters. Input data have to satisfy both conditions.")
    cols = raster_red.RasterXSize
    rows = raster_red.RasterYSize
    final_mask = np.zeros(shape=(rows, cols), dtype='uint8')
    final_pts = []
#pocetak for petlje za masku
    for i in range(len(lyr)):
        feat = lyr.GetFeature(i)
        geom = feat.GetGeometryRef()

        #dims
        geo_transform = raster_red.GetGeoTransform()
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

        final_pts.append(pts)
        mask = np.zeros(shape=(rows, cols), dtype='uint8')
        cv2.fillPoly(mask, [pts], 255)
        mask = (mask[:, :] == 255)
        final_mask = final_mask+mask
        print("Buildovana maska za layer broj : " + str(i))

    final_final_pts = []
    for i in range(len(final_pts)):
        if i ==0:
            final_final_pts = np.concatenate([np.reshape(final_final_pts,[0,2]),final_pts[i]])
        else:
            final_final_pts = np.concatenate([final_final_pts, final_pts[i]])
    final_mask[final_mask==2] = 0
#kraj for petlje
    # Bounding box of non-black pixels.
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

    rr = raster_red.GetRasterBand(1)
    rg = raster_green.GetRasterBand(1)
    rb = raster_blue.GetRasterBand(1)
    rnir = raster_nir.GetRasterBand(1)
    rre = raster_rededge.GetRasterBand(1)
    # Ucitavanje kanala slike # 3 seconds
    ch_red = rr.ReadAsArray()
    ch_green = rg.ReadAsArray()
    ch_blue = rb.ReadAsArray()
    ch_nir = rnir.ReadAsArray()
    ch_rededge = rre.ReadAsArray()

    ch_red_croped = copy.deepcopy(ch_red[y0:y1, x0:x1])
    ch_green_croped = copy.deepcopy(ch_green[y0:y1, x0:x1])
    ch_blue_croped = copy.deepcopy(ch_blue[y0:y1, x0:x1])
    ch_nir_croped = copy.deepcopy(ch_nir[y0:y1,x0:x1]).astype("float32")
    ch_rededge_croped = copy.deepcopy(ch_rededge[y0:y1,x0:x1]).astype("float32")

    return (ch_blue_croped, ch_green_croped, ch_red_croped, ch_nir_croped, 
            ch_rededge_croped, cropped_mask, pixelWidth, pixelHeight, grid_raster_filename_red, 
            grid_raster_filename_green, grid_raster_filename_blue, grid_raster_filename_nir, 
            grid_raster_filename_rededge, y0, x0, y1, x1)


def index_calculation_v2(in_R, in_G, in_B, in_NIR, in_RedEdge, mask_ROI):
    mask_ROI = mask_ROI.astype('bool')
    TGI = np.zeros(shape=[in_R.shape[0], in_R.shape[1]], dtype='float64')
    GLI = np.zeros(shape=[in_R.shape[0], in_R.shape[1]], dtype='float64')
    ExG = np.zeros(shape=[in_R.shape[0], in_R.shape[1]], dtype='float64')
    CIVE = np.zeros(shape=[in_R.shape[0], in_R.shape[1]], dtype='float64')
    NDVI = np.zeros(shape=[in_R.shape[0], in_R.shape[1]], dtype='float64')
    WDRVI = np.zeros(shape=[in_R.shape[0], in_R.shape[1]], dtype='float64')
    CCCI = np.zeros(shape=[in_R.shape[0], in_R.shape[1]], dtype='float64')
    CIrededge = np.zeros(shape=[in_R.shape[0], in_R.shape[1]], dtype='float64')
    MGRV = np.zeros(shape=[in_R.shape[0], in_R.shape[1]], dtype='float64')
    MPRI = np.zeros(shape=[in_R.shape[0], in_R.shape[1]], dtype='float64')
    RGBVI = np.zeros(shape=[in_R.shape[0], in_R.shape[1]], dtype='float64')

    R_roi = copy.deepcopy(in_R[mask_ROI == True]).astype('float64')
    G_roi = copy.deepcopy(in_G[mask_ROI == True]).astype('float64')
    B_roi = copy.deepcopy(in_B[mask_ROI == True]).astype('float64')
    NIR_roi = copy.deepcopy(in_NIR[mask_ROI == True]).astype('float64')
    RedEdge_roi = copy.deepcopy(in_RedEdge[mask_ROI == True]).astype('float64')

    TGI_roi = (TGI[mask_ROI == True]).view()
    GLI_roi = (GLI[mask_ROI == True]).view()
    ExG_roi = (ExG[mask_ROI == True]).view()
    CIVE_roi = (CIVE[mask_ROI == True]).view()
    NDVI_roi = (NDVI[mask_ROI == True]).view()
    WDRVI_roi = (WDRVI[mask_ROI == True]).view()
    CCCI_roi = (CCCI[mask_ROI == True]).view()
    CIrededge_roi = (CIrededge[mask_ROI == True]).view()
    MGRV_roi = (MGRV[mask_ROI == True]).view()
    MPRI_roi = (MPRI[mask_ROI == True]).view()
    RGBVI_roi = (RGBVI[mask_ROI == True]).view()

    max_R = np.max(R_roi)
    max_G = np.max(G_roi)
    max_B = np.max(B_roi)
    max_NIR = np.max(NIR_roi)
    max_RedEdge = np.max(RedEdge_roi)
    max_const = (max(max_R, max_G, max_B,max_NIR,max_RedEdge)).astype("float64")
    alpha=0.1
    calculation_of_TGI_and_GLI(R_roi, G_roi, B_roi, TGI_roi, GLI_roi, max_const)
    calculation_of_WDRVI_and_CCCI_and_NDVI_and_CIrededge(R_roi, NIR_roi, RedEdge_roi,alpha,NDVI_roi,WDRVI_roi,CCCI_roi,CIrededge_roi)
    calculation_of_MGRV_and_MPRI_and_RGBVI(R_roi,G_roi,B_roi,MGRV_roi,MPRI_roi,RGBVI_roi)

    R_roi[:] /= max_R
    G_roi[:] /= max_G
    B_roi[:] /= max_B

    sum_RGB = R_roi + G_roi + B_roi + 1.0

    R_roi[:] /= sum_RGB
    G_roi[:] /= sum_RGB
    B_roi[:] /= sum_RGB

    calculation_of_ExG_and_CIVE(R_roi, G_roi, B_roi, ExG_roi, CIVE_roi)

    TGI[mask_ROI == True] = TGI_roi
    GLI[mask_ROI == True] = GLI_roi
    ExG[mask_ROI == True] = ExG_roi
    CIVE[mask_ROI == True] = CIVE_roi
    NDVI[mask_ROI == True] = NDVI_roi
    WDRVI[mask_ROI == True] = WDRVI_roi
    MGRV[mask_ROI == True] = MGRV_roi
    MPRI[mask_ROI == True] = MPRI_roi
    RGBVI[mask_ROI == True] = RGBVI_roi

    return  ExG, CIVE, TGI, GLI, NDVI, WDRVI, MGRV, MPRI, RGBVI

def vegetation_detection_v2(WDRVI, MGRV, mask_ROI):
    mask_ROI = mask_ROI.astype('bool')
    #global otsu thresholding
    thresh_WDRVI = threshold_otsu(WDRVI[mask_ROI == True])
    thresh_MGRV = threshold_otsu(MGRV[mask_ROI == True])  # thresh_VARI = threshold_otsu(VARI[mask_ROI.astype("bool")])

    binary_WDRVI = (WDRVI > thresh_WDRVI).view()
    binary_MGRV = (MGRV > thresh_MGRV).view()  # binary_VARI = VARI > thresh_VARI

    vegetation_mask = (binary_WDRVI | binary_MGRV) & mask_ROI
    return vegetation_mask

def clusterization_of_vegetation_coverage_v2(ExG, NDVI, WDRVI, MGRV, MPRI, RGBVI, vegetation_mask, n_cluster_in):
    indices = np.nonzero(vegetation_mask)
    samples = np.column_stack(
        [NDVI[vegetation_mask == True], WDRVI[vegetation_mask == True],
         MGRV[vegetation_mask == True], MPRI[vegetation_mask == True], RGBVI[vegetation_mask == True]])

    scaler = StandardScaler()
    scaler.fit(samples)
    dataset = scaler.transform(samples)

    b_size =  int(0.2 *(samples.shape[0] / 100)) # 0.2%

    # clusterization
    with joblib.parallel_backend("threading"):
        kmeans_clusterization = MiniBatchKMeans(n_clusters=n_cluster_in, batch_size= b_size, random_state=0).fit(dataset)

    # visualization
    labels_kmeans = kmeans_clusterization.labels_
    label_image_1d_kmeans = np.zeros(shape=(ExG.shape[0], ExG.shape[1]), dtype="uint8")
    for i in range(n_cluster_in):
        label_image_1d_kmeans[indices[0][labels_kmeans == i], indices[1][labels_kmeans == i]] = colors_grey[i]

    return label_image_1d_kmeans


def distance_two_points(p1,p2, dist_type):

    dist = 0.0
    if dist_type == 'Euclidian':
        dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    elif dist_type =='Manhattan':
        dist = np.abs(p1[0] - p2[0]) + np.abs((p1[1] - p2[1]))

    return dist

def prob_map_calculation(input_map, step):

    working_input_map = np.pad(input_map, (step, step), 'edge')
    prob_map = np.zeros(shape=input_map.shape,dtype="float64")
    integral_image = cv2.integral(working_input_map,sdepth=cv2.CV_64F)
    prob_map_calc(prob_map,integral_image,step)

    return prob_map

def generate_sampling_points_v3(label_map_array,skelet_map):

    # Add two more parameters
    # one defines number of points per zone (cluster)
    # second minimum distance between sampling points within cluster.
    # check rounding of float values for pixel coordinates.

    sampling_points = [] #(add first x and then y)
    for c in range(label_map_array.shape[2]):

        binary_image = (label_map_array[:, :, c] != 0).astype("uint8")
        binary_image[:,:400] = 0
        binary_image[:, -400:] = 0
        binary_image[:400,:] = 0
        binary_image[-400:,:] = 0

        connectivity = 8
        num_labels_start, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity, cv2.CV_32S)

        for i in range(1, num_labels_start):

            component_mask = copy.deepcopy(labels == i)

            dist = cv2.distanceTransform(component_mask.astype("uint8"), cv2.DIST_L2, 3)*skelet_map
            min_value = np.min(dist[component_mask])
            max_value = np.max(dist[component_mask])
            if min_value == max_value:
                if max_value < 50:
                    continue
                else:
                    sampling_points.append(list(np.hstack([np.round(centroids[i,:]).astype("int32"), 1])))
            else:
                if max_value < 50:
                    continue
                else:
                    [y, x] = np.nonzero(dist)
                    data = np.column_stack([y,x,dist[y,x]])
                    sorted_ind = np.argsort(data[:,2],axis=0)[::-1]
                    sampling_points.append([np.round(data[sorted_ind[0],1]).astype("int32"),np.round(data[sorted_ind[0],0]).astype("int32"),1])

    return sampling_points

def generate_decision_zones_map_with_sampling_points(input_map, 
                                                     mask_ROI, 
                                                     num_zones, 
                                                     window_size,
                                                     y_start, 
                                                     x_start,  
                                                     mask_ROI_tmp):
    M = np.zeros(shape=[2, 3], dtype="float64")
    y_offset = 0
    x_offset = 0

    mask_ROI_extended_rot = np.zeros(shape=mask_ROI.shape, dtype='uint8')
    mask_ROI_extended_rot2 = np.zeros(shape=mask_ROI_tmp.shape, dtype='uint8')
    input_map_extended_rot = np.zeros(shape=input_map.shape, dtype='uint8')

    contours, hierarchy = cv2.findContours(mask_ROI_tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rotrect = cv2.minAreaRect(contours[0])
    box = cv2.boxPoints(rotrect)
    box = np.int0(box)
    cv2.drawContours(mask_ROI_tmp, [box], 0, 255, -1)

    # logic to find three points for rotation
    print("logic to find three points for rotation")
    dist_euclid = np.sqrt(np.sum(np.square(box - np.array([0, 0])), axis=1))
    ind_min = np.argmin(dist_euclid)
    if ind_min == 0:
        ind_previous = box.shape[0] - 1
        ind_next = ind_min + 1
    elif ind_min == (box.shape[0] - 1):
        ind_previous = ind_min - 1
        ind_next = 0
    else:
        ind_previous = ind_min - 1
        ind_next = ind_min + 1

    pts_0 = complex(box[ind_previous][0], box[ind_previous][1])
    pts_1 = complex(box[ind_next][0], box[ind_next][1])
    r_0, phi_0 = cmath.polar(pts_0)
    r_1, phi_1 = cmath.polar(pts_1)

    if phi_0 >= phi_1:
        pts1 = np.float32([box[ind_previous], box[ind_min], box[ind_next]])
        pts2 = np.float32([[0, int(distance_two_points(box[ind_min], box[ind_previous], 'Euclidian'))], 
                           [0, 0], 
                           [int(distance_two_points(box[ind_min], box[ind_next], 'Euclidian')), 0]])
    else:
        pts1 = np.float32([box[ind_next], box[ind_min], box[ind_previous]])
        pts2 = np.float32([[0, int(distance_two_points(box[ind_min], box[ind_next], 'Euclidian'))], 
                           [0, 0], 
                           [int(distance_two_points(box[ind_min], box[ind_previous], 'Euclidian')), 0]])

    M[:, :] = cv2.getAffineTransform(pts1, pts2)
    invert_M = cv2.invertAffineTransform(M)
    mask_ROI_extended_rot[:, :] = cv2.warpAffine(mask_ROI, M, (mask_ROI.shape[1], mask_ROI.shape[0]))
    mask_ROI_extended_rot2[:, :] = cv2.warpAffine(mask_ROI_tmp, M, (mask_ROI_tmp.shape[1], mask_ROI_tmp.shape[0]))
    mask_ROI_extended_rot = mask_ROI_extended_rot.astype("bool")
    mask_ROI_extended_rot2 = mask_ROI_extended_rot2.astype("bool")

    mask_map = np.zeros(shape=mask_ROI_extended_rot2.shape, dtype="bool")
    mask_map[:,:] = False
    for i in range(num_zones):
        rot_dec_map = cv2.warpAffine((input_map == colors_grey[i]).astype("uint8"), M, (input_map.shape[1], input_map.shape[0]))
        label_ROI = ~mask_map[:,:] & rot_dec_map.astype("bool") & mask_ROI_extended_rot[:,:]
        input_map_extended_rot[label_ROI.astype("bool")] = colors_grey[i]
        mask_map[:, :] = (input_map_extended_rot != 0) & mask_ROI_extended_rot[:,:]

    [y, x] = np.nonzero(mask_ROI_extended_rot)

    x_extent = np.max(x)
    y_extent = np.max(y)

    prob_map = np.zeros(shape=(y_extent, x_extent, num_zones), dtype="float64")
    label_map_array = np.zeros(shape=(y_extent, x_extent, num_zones), dtype="float64")

    print("prob map caluclation")
    for i in range(num_zones):
        prob_map[:, :, i] = prob_map_calculation((input_map_extended_rot[0:y_extent, 0:x_extent] == colors_grey[i]).astype('uint8'), 
                                                 int(window_size / 2))

    max_prob_map = np.argmax(prob_map, axis=2)
    for i in range(num_zones):
        max_prob_map[max_prob_map == i] = colors_grey[i]
    max_prob_map = max_prob_map.astype('uint8')

    dec_map = np.zeros(shape=[input_map_extended_rot.shape[0], input_map_extended_rot.shape[1]], dtype='uint8')
    zone_map = np.zeros(shape=[input_map_extended_rot.shape[0], input_map_extended_rot.shape[1]], dtype='uint8')
    dec_map[0:y_extent, 0:x_extent] = copy.deepcopy(max_prob_map)

    for i in range(num_zones):
        label_map_array[:, :, i] = (dec_map[0:y_extent, 0:x_extent] == colors_grey[i]).astype("uint8")
    
    print("generate sampling points")
    rot_sampling_points = generate_sampling_points_v3(label_map_array, mask_map[:y_extent, :x_extent])
    rot_sampling_points = np.asarray(rot_sampling_points)
    sampling_points = []
    
    for i in range(rot_sampling_points.shape[0]):
        temp_point = invert_M @ rot_sampling_points[i,:]
        temp_point[0] = temp_point[0] - x_offset
        temp_point[1] = temp_point[1] - y_offset
        sampling_points.append((temp_point).astype("int64"))

    sampling_points = np.asarray(sampling_points)
    final_dec_map = np.zeros(shape=mask_ROI.shape,dtype="uint8")
    mask_map = np.zeros(shape=mask_ROI.shape, dtype="bool")
    mask_map[:,:] = False

    ind_array_list = []
    for i in range(num_zones):
        dec_map_irot = cv2.warpAffine((dec_map == colors_grey[i]).astype("uint8"), invert_M, (input_map_extended_rot.shape[1], input_map_extended_rot.shape[0]))
        tmp_map = dec_map_irot[y_offset:y_offset + mask_ROI.shape[0],x_offset:x_offset + mask_ROI.shape[1]]
        label_ROI = (~mask_map[:,:] & tmp_map.astype("bool") & mask_ROI[:,:]).astype("bool")
        if np.any(label_ROI):
            final_dec_map[label_ROI.astype("bool")] = colors_grey[i]
            mask_map[:, :] = (final_dec_map != 0) & mask_ROI[:,:]
            ind_array_list.append(i)

    final_dec_map[mask_ROI == False] = 0
    ind_array = np.arange(len(ind_array_list))

    map_index = {}
    counter = 0
    for keys in ind_array_list:
        map_index[keys] = ind_array[counter]
        counter += 1

    #check for sampling points
    final_sampling_points = []
    y_mask, x_mask = np.where(mask_ROI != 0)
    for i in range(sampling_points.shape[0]):
        if (sampling_points[i,1] < final_dec_map.shape[0]) and (sampling_points[i,0] < final_dec_map.shape[1]):
            if final_dec_map[sampling_points[i,1],sampling_points[i,0]] != 0:
                sampling_points[i, 1] = sampling_points[i, 1] + y_start
                sampling_points[i, 0] = sampling_points[i, 0] + x_start
                final_sampling_points.append(sampling_points[i].tolist())
            else:
                sampling_points[i, 1] = sampling_points[i, 1] + y_start
                sampling_points[i, 0] = sampling_points[i, 0] + x_start
                final_sampling_points.append(sampling_points[i].tolist())
    final_sampling_points = np.asarray(final_sampling_points)

    return final_dec_map, ind_array, map_index, final_sampling_points

def sampling_points_generator(in_raster, sampling_points, outShapefile):

    raster = gdal.Open(in_raster)
    srs_cs = osr.SpatialReference()
    srs_cs.ImportFromWkt(raster.GetProjectionRef())
    geo_transform = raster.GetGeoTransform()
    xOrigin = geo_transform[0]
    yOrigin = geo_transform[3]
    pixelWidth = geo_transform[1]
    pixelHeight = geo_transform[5]

    # outShapefile = out_shape
    outDriver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(outShapefile):
        outDriver.DeleteDataSource(outShapefile)

    outDataSource = outDriver.CreateDataSource(outShapefile)
    outLayer = outDataSource.CreateLayer("zones", geom_type=ogr.wkbPoint, srs=srs_cs)
    outLayer.CreateField(ogr.FieldDefn('ID', ogr.OFTInteger))
    outLayer.CreateField(ogr.FieldDefn('X_coord',ogr.OFTReal))
    outLayer.CreateField(ogr.FieldDefn('Y_coord',ogr.OFTReal))

    id_num = 1
    x_coord_vec = []
    y_coord_vec = []

    for i in range(sampling_points.shape[0]):

        X_coord = xOrigin + pixelWidth * sampling_points[i,0]# + pixelWidth * 0.5
        Y_coord = yOrigin + pixelHeight * sampling_points[i,1]# + pixelHeight * 0.5
        x_coord_vec.append(X_coord)
        y_coord_vec.append(Y_coord)
        # addpoint to shapefile
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(X_coord, Y_coord)

        outLayerDefn = outLayer.GetLayerDefn()
        new_feature = ogr.Feature(outLayerDefn)
        new_feature.SetGeometry(point)
        new_feature.SetField('ID', int(id_num))
        new_feature.SetField('X_coord', float(X_coord))
        new_feature.SetField('Y_coord', float(Y_coord))

        outLayer.CreateFeature(new_feature)
        new_feature = None

        id_num += 1

class Map_stat(object):

    #constructor
    def __init__(self, soil_percentage, vegetation_percentage, cl_percentage, cl_indices_mean,cl_indices_std):

        self.soil_percentage = soil_percentage
        self.vegetation_percentage = vegetation_percentage
        self.cl_percentage = cl_percentage
        self.cl_indices_mean = cl_indices_mean
        self.cl_indices_std = cl_indices_std


def statistics_from_map(label_image_1d_kmeans, ind_list, map_index, veg_mask, mask_ROI, ExG, CIVE, TGI, GLI):

    cl_mask = []
    cl_veg = []
    cl_soil = []
    consider_pixel_label = []
    soil_percent = []
    veg_percent = []
    cl_percentage = []
    vegetation_pixels = np.count_nonzero(veg_mask)
    consider_pixels_all = np.count_nonzero(mask_ROI)
    soil_pixels = consider_pixels_all - vegetation_pixels
    if soil_pixels == 0:
        soil_pixels = 1.0

    for i in range(len(ind_list)):
        field = list(map_index.keys())[i]
        cl_mask.append(label_image_1d_kmeans == colors_grey[field])#append(label_image_1d_kmeans == colors_grey[ind_list[date]])
        cl_veg.append(np.count_nonzero(cl_mask[-1] & veg_mask))
        cl_soil.append(np.count_nonzero(cl_mask[-1] & ~veg_mask))
        consider_pixel_label.append(np.count_nonzero(cl_mask[-1]))
        soil_percent.append(cl_soil[-1] / soil_pixels)
        veg_percent.append(cl_veg[-1] / vegetation_pixels)
        cl_percentage.append(consider_pixel_label[-1] / consider_pixels_all)

    cl_indices_mean = []
    cl_indices_std = []
    with joblib.parallel_backend("threading"):
        for i in range(len(ind_list)):
            cl_indices_mean.append(np.array([np.mean(ExG[cl_mask[i]]),np.mean(CIVE[cl_mask[i]]),np.mean(TGI[cl_mask[i]]),np.mean(GLI[cl_mask[i]])]))
            cl_indices_std.append(np.array([np.std(ExG[cl_mask[i]]), np.std(CIVE[cl_mask[i]]), np.std(TGI[cl_mask[i]]), np.std(GLI[cl_mask[i]])]))

    calc_stats = Map_stat(soil_percent,veg_percent,cl_percentage,cl_indices_mean,cl_indices_std)

    return calc_stats

def create_vegetation_index_map(in_raster, 
                                        veg_ind,
                                        out_pth,
                                        idx_tag,
                                        y_start, 
                                        y_end, 
                                        x_start, 
                                        x_end):

    
    srsProj = osr.SpatialReference()
    srsProj.ImportFromEPSG(32634)
    raster = gdal.Open(in_raster)
    rows = raster.RasterYSize
    cols = raster.RasterXSize
    final_mask = np.zeros(shape=[rows,cols],dtype="float32")
    print("final_mask shape: ",final_mask.shape)
    print("veg_ind shape: ",veg_ind.shape)
    final_mask[y_start:y_end, x_start:x_end] = copy.deepcopy(veg_ind)
    driver = gdal.GetDriverByName('GTiff')
    tmp_raster = driver.Create(os.path.join(out_pth,'test_veg_ind_'+idx_tag+'.tif'), final_mask.shape[1], final_mask.shape[0], 1, gdal.GDT_Float32)
    tmp_raster.SetGeoTransform(raster.GetGeoTransform())
    tmp_raster.SetProjection(raster.GetProjectionRef())
    srcband = tmp_raster.GetRasterBand(1)
    srcband.WriteArray(final_mask)
    srcband.FlushCache()
    
    
    

def create_shp_multipolygon_from_raster(in_raster, 
                                        ind_list, 
                                        map_index, 
                                        image, 
                                        out_shape_tmp, 
                                        out_shape, 
                                        calc_stats, 
                                        y_start, 
                                        y_end, 
                                        x_start, 
                                        x_end):

    srsProj = osr.SpatialReference()
    srsProj.ImportFromEPSG(32634)

    raster = gdal.Open(in_raster)
    rows = raster.RasterYSize
    cols = raster.RasterXSize
    new_image = np.zeros(shape=[rows,cols],dtype="uint8")
    # new_image[y_start:y_end,x_start:x_end] = copy.deepcopy(image)
    # row_pad = 1035 # parametar paddinga
    # column_pad = 1288 # parametar paddinga
    new_image[y_start:y_end, x_start:x_end] = copy.deepcopy(image)
    driver = gdal.GetDriverByName('GTiff')
    tmp_raster = driver.Create('test.tif', new_image.shape[1], new_image.shape[0], 1, gdal.GDT_Byte)
    tmp_raster.SetGeoTransform(raster.GetGeoTransform())
    tmp_raster.SetProjection(raster.GetProjectionRef())
    srcband = tmp_raster.GetRasterBand(1)
    srcband.WriteArray(new_image)
    srcband.FlushCache()

    geometries = []
    union = []
    for i in range(len(ind_list)):
         geometries.append(ogr.Geometry(ogr.wkbMultiPolygon))

    outShapefile = out_shape_tmp
    outDriver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(outShapefile):
        outDriver.DeleteDataSource(outShapefile)

    outDataSource = outDriver.CreateDataSource(outShapefile)
    outLayer = outDataSource.CreateLayer("zones", geom_type=ogr.wkbMultiPolygon, srs=srsProj)
    outLayer.CreateField(ogr.FieldDefn('ID', ogr.OFTInteger))
    outLayer.CreateField(ogr.FieldDefn('SOIL', ogr.OFTReal))
    outLayer.CreateField(ogr.FieldDefn('VEG', ogr.OFTReal))
    outLayer.CreateField(ogr.FieldDefn('CL', ogr.OFTReal))
    outLayer.CreateField(ogr.FieldDefn('ExG_m', ogr.OFTReal))
    outLayer.CreateField(ogr.FieldDefn('CIVE_m', ogr.OFTReal))
    outLayer.CreateField(ogr.FieldDefn('TGI_m', ogr.OFTReal))
    outLayer.CreateField(ogr.FieldDefn('GLI_m', ogr.OFTReal))
    outLayer.CreateField(ogr.FieldDefn('ExG_s', ogr.OFTReal))
    outLayer.CreateField(ogr.FieldDefn('CIVE_s', ogr.OFTReal))
    outLayer.CreateField(ogr.FieldDefn('TGI_s', ogr.OFTReal))
    outLayer.CreateField(ogr.FieldDefn('GLI_s', ogr.OFTReal))
    options = ['8CONNECTED=8']

    result = gdal.Polygonize(srcband, None, outLayer, 0, options, callback=None)
    outLayer.ResetReading()

    for feature in outLayer:
        if feature.GetField('ID') != 0:
            index = np.where(colors_grey == feature.GetField('ID'))[0][0]
            ind = map_index[index]
            geom = feature.GetGeometryRef()

            if geom.GetGeometryName() == 'MULTIPOLYGON':
                for geom_part in geom:
                    geometries[ind].AddGeometry(geom_part.SimplifyPreserveTopology(0.05))
            else:
                geometries[ind].AddGeometry(geom.SimplifyPreserveTopology(0.05))

    for i in range(len(ind_list)):
        union.append(geometries[i].UnionCascaded())
        # print(geometries[i].UnionCascaded())

    outShapefile_new = out_shape
    outDriver_new = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(outShapefile_new):
        outDriver_new.DeleteDataSource(outShapefile_new)

    outDataSource_new = outDriver_new.CreateDataSource(outShapefile_new)
    outLayer_new = outDataSource_new.CreateLayer("zones_new", geom_type=ogr.wkbMultiPolygon, srs=srsProj)
    outLayer_new.CreateField(ogr.FieldDefn('ID', ogr.OFTInteger))
    outLayer_new.CreateField(ogr.FieldDefn('SOIL', ogr.OFTReal))
    outLayer_new.CreateField(ogr.FieldDefn('VEG', ogr.OFTReal))
    outLayer_new.CreateField(ogr.FieldDefn('CL', ogr.OFTReal))
    outLayer_new.CreateField(ogr.FieldDefn('ExG_m', ogr.OFTReal))
    outLayer_new.CreateField(ogr.FieldDefn('CIVE_m', ogr.OFTReal))
    outLayer_new.CreateField(ogr.FieldDefn('TGI_m', ogr.OFTReal))
    outLayer_new.CreateField(ogr.FieldDefn('GLI_m', ogr.OFTReal))
    outLayer_new.CreateField(ogr.FieldDefn('ExG_s', ogr.OFTReal))
    outLayer_new.CreateField(ogr.FieldDefn('CIVE_s', ogr.OFTReal))
    outLayer_new.CreateField(ogr.FieldDefn('TGI_s', ogr.OFTReal))
    outLayer_new.CreateField(ogr.FieldDefn('GLI_s', ogr.OFTReal))

    ind = int(0)
    for i in range(len(ind_list)):
        outLayer_new = add_feature_to_shapefile(outLayer_new, union[i], ind, calc_stats)
        ind = ind + 1

    outDataSource_new = None


def add_feature_to_shapefile(outLayer,geometry,index,calc_stats):

    outLayerDefn = outLayer.GetLayerDefn()
    new_feature = ogr.Feature(outLayerDefn)
    new_feature.SetGeometry(geometry)
    new_feature.SetField('ID', int(index))
    new_feature.SetField('SOIL', calc_stats.soil_percentage[index])
    new_feature.SetField('VEG', calc_stats.vegetation_percentage[index])
    new_feature.SetField('CL', calc_stats.cl_percentage[index])
    new_feature.SetField('ExG_m', calc_stats.cl_indices_mean[index][0])
    new_feature.SetField('CIVE_m', calc_stats.cl_indices_mean[index][1])
    new_feature.SetField('TGI_m', calc_stats.cl_indices_mean[index][2])
    new_feature.SetField('GLI_m', calc_stats.cl_indices_mean[index][3])
    new_feature.SetField('ExG_s', calc_stats.cl_indices_std[index][0])
    new_feature.SetField('CIVE_s', calc_stats.cl_indices_std[index][1])
    new_feature.SetField('TGI_s', calc_stats.cl_indices_std[index][2])
    new_feature.SetField('GLI_s', calc_stats.cl_indices_std[index][3])
    outLayer.CreateFeature(new_feature)
    new_feature = None

    return outLayer