# Vegetation index maps calculator - Inference service

The pipeline consists of three components or blocks. Through this pipeline, as in all previous pipelines, we must firstly download data from the data storage (minIO), then we have to choose a part of parcel where we want to calculate VI maps and lastly provide the maps back to our data storage in tif or any other suitable format.

#### Step 1: Creating notebook

The user can create new notebook for this task, as explained in [here](https://github.com/FlexiGroBots-H2020/AI-platform/blob/c07ef85224c4533fd04f80b07a5ba4398e17597c/kubeflow/Documentation.md#3-pipeline-generation) , or use already existing. 

#### Step 2: Uploading yaml files for each component

YAML files for this pipeline can be found in FlexiGroBots [Github](https://github.com/FlexiGroBots-H2020/AI-platform/tree/c07ef85224c4533fd04f80b07a5ba4398e17597c/kubeflow/Vegetation%20indices%20calc) repo in the corresponding folders. Each file assigns appropriate docker image/package:
- Download component: rebuild_delineation_download_component 
- Test component: vegetation_ind_calc_component
- PostProcessing component: rebuild_delineation_postprocessing_component.

These docker images were pushed to the project GitHub repo and can be found in [packages](https://github.com/orgs/FlexiGroBots-H2020/packages) section.

An IPYNB file is also necessary and it is included in the same folders in GitHub repo. This file is essential for connecting individual components. It consists of at least two major cells. In the first one it’s always important to check volume name in the code and change it to match with corresponding volume:

![notebook_volume_name](https://github.com/Dimitrije2507/BlueberryRowDetectionKubeflow/blob/c94f5e0ece48688947c913a2e655ff163881c747/notebook_volume_name.png)


After that, the user should initialize each pipeline component by loading them from corresponding yaml files: 

```python
download_component = kfp.components.load_component_from_file('Vegetation_indices_calculation/download_model.yaml')
veg_ind_model = kfp.components.load_component_from_file('Vegetation_indices_calculation/veg_index_model.yaml')
postprocessing_model = kfp.components.load_component_from_file('Vegetation_indices_calculation/postprocessing_model.yaml')
```
Initialization of download component:
```python
download_component_task = download_component(
        "biosens-test-inference-service", # minIO bucket where the input data is stored
        "/mnt/InferenceData/" # Folder within the notebook/volume where the data will be downloaded
        ).add_pvolumes({"/mnt": vop})
download_component_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
```
Initialization of VI calculator component:

```python
veg_ind_model_task = veg_ind_model(
        4, # Number of zones
        10.0, # window size in meters
        "/mnt/InferenceData/DataTest/shp/TestParcel_Belanovica1.shp", # Shape file of test parcel
        "/mnt/InferenceData/DataTest/GeoTiffs/Belanovica1_processed_transparent_mosaic_red.tif", # red band 
        "/mnt/InferenceData/DataTest/GeoTiffs/Belanovica1_processed_transparent_mosaic_green.tif", # green band
        "/mnt/InferenceData/DataTest/GeoTiffs/Belanovica1_processed_transparent_mosaic_blue.tif", # blue band
        "/mnt/InferenceData/DataTest/GeoTiffs/Belanovica1_processed_transparent_mosaic_nir.tif", # nir band
        "/mnt/InferenceData/DataTest/GeoTiffs/Belanovica1_processed_transparent_mosaic_red edge.tif", # red edge band
        "/mnt/InferenceData/DataTest/vegetation_index_calc_results/run1/", # destination folder for saving results
        download_component_task.output).add_pvolumes({"/mnt": vop})
veg_ind_model_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
```
Initialization of postprocessing component used for uploading results to minIO:

```python
postprocessing_model = postprocessing_model(
    "blueberry-results", # minIO bucket where the resulting VI maps will be uploaded
    "/mnt/InferenceData/DataTest/vegetation_index_calc_results/run1/TEMP", # folder within the notebook/volume where the results that should be uploaded are stored. 
    "TEMP", # name of the folder within the minIO bucket where the data will be uploaded
    veg_ind_model_task.output
    ).add_pvolumes({"/mnt": vop})
postprocessing_model.execution_options.caching_strategy.max_cache_staleness = "P0D"
```

Running the pipeline initialization:

```python
if __name__ == '__main__':
    file_name = "Vegetation_indices_calculation/full_vegetation_index_pipeline.yaml"
    kfp.compiler.Compiler().compile(full_pipeline, file_name)
```

Uploading custom pipeline to Kubeflow Pipelines section by running cell:

```python
import kfp
import random
client=kfp.Client()
filepath = '/home/jovyan/delineation-and-zoning-datavol-1/' + file_name
name = 'VegetationIndexCalculationPipeline_timestamp'
print("Uploaded pipeline:", name)
pipeline = client.pipeline_uploads.upload_pipeline(filepath, name=name)
```
New pipeline will be created in Kubflow pipelines section and by creating new run as explained in [here](https://github.com/FlexiGroBots-H2020/AI-platform/blob/c07ef85224c4533fd04f80b07a5ba4398e17597c/kubeflow/Documentation.md#3-pipeline-generation)


