# Delineation and zoning - Inference service

The pipeline consists of three components or blocks. Again, we must first download data from the data storage (minIO), then we have to choose a part of parcel where we want to calculate vegetation indices and generate maps that are further being used in ML pipeline for delineation and zoning. Lastly we need to provide the parcel zones back to our data storage in tif, shp or any other suitable format.

#### Step 1: Creating notebook

The user can create new notebook for this task, as explained in [here](https://github.com/FlexiGroBots-H2020/AI-platform/blob/c07ef85224c4533fd04f80b07a5ba4398e17597c/kubeflow/Documentation.md#3-pipeline-generation) , or use already existing. 

#### Step 2: Uploading yaml files for each component

YAML files for this pipeline can be found in FlexiGroBots [Github](https://github.com/FlexiGroBots-H2020/AI-platform/tree/biosens/kubeflow/Delineation%20Inference) repo in the corresponding folders under the biosens branch. Each file assigns appropriate docker image/package:
- Download component: rebuild_delineation_download_component 
- Test component: rebuild_delineation_test_component
- PostProcessing component: rebuild_delineation_postprocessing_component.

These docker images were pushed to the project GitHub repo and can be found in [packages](https://github.com/orgs/FlexiGroBots-H2020/packages) section.

An IPYNB file is also necessary and it is included in the same folders in GitHub repo. This file is essential for connecting individual components. It consists of at least two major cells. In the first one it’s always important to check volume name in the code and change it to match with corresponding volume:

![notebook_volume_name](https://github.com/Dimitrije2507/BlueberryRowDetectionKubeflow/blob/c94f5e0ece48688947c913a2e655ff163881c747/notebook_volume_name.png)

After that, the user should initialize each pipeline component by loading them from corresponding yaml files:
```python
download_component = kfp.components.load_component_from_file('download_model.yaml')
zoning_model = kfp.components.load_component_from_file('zoning_model.yaml')
postprocessing_model = kfp.components.load_component_from_file('postprocessing_model.yaml')
```

Initialization of download component:
```python
download_component_task = download_component(
        "biosens-test-inference-service",
        # "biosens-test2",
        "/mnt/InferenceData/"
        ).add_pvolumes({"/mnt": vop})
download_component_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
```

Initialization of Delineation and zoning component:
```python
zoning_model_task = zoning_model(
        4,
        10.0,
        "/mnt/InferenceData/DataTest/shp/TestParcel_Belanovica1.shp",
        # "/mnt/InferenceData/DataTest/shp/test_parcela_shape.shp",
        "/mnt/InferenceData/DataTest/GeoTiffs/Belanovica1_processed_transparent_mosaic_red.tif",
        # "/mnt/InferenceData/DataTest/GeoTiffs/Babe_registrated_corrected_transparent_mosaic_red.tif",
        "/mnt/InferenceData/DataTest/GeoTiffs/Belanovica1_processed_transparent_mosaic_green.tif",
        # "/mnt/InferenceData/DataTest/GeoTiffs/Babe_registrated_corrected_transparent_mosaic_green.tif",
        "/mnt/InferenceData/DataTest/GeoTiffs/Belanovica1_processed_transparent_mosaic_blue.tif",
        # "/mnt/InferenceData/DataTest/GeoTiffs/Babe_registrated_corrected_transparent_mosaic_blue.tif",
        "/mnt/InferenceData/DataTest/GeoTiffs/Belanovica1_processed_transparent_mosaic_nir.tif",
        # "/mnt/InferenceData/DataTest/GeoTiffs/Babe_registrated_corrected_transparent_mosaic_nir.tif",
        "/mnt/InferenceData/DataTest/GeoTiffs/Belanovica1_processed_transparent_mosaic_red edge.tif",
        # "/mnt/InferenceData/DataTest/GeoTiffs/Babe_registrated_corrected_transparent_mosaic_red edge.tif",
        "/mnt/InferenceData/DataTest/zoning_results/output_files/",
        download_component_task.output).add_pvolumes({"/mnt": vop})
        # "/mnt/InferenceData/DataTest/").add_pvolumes({"/mnt": vop})
zoning_model_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
```
Initialization of postprocessing component used for uploading results to minIO:
```python
postprocessing_model = postprocessing_model(
        "blueberry-results",
        "/mnt/InferenceData/DataTest/zoning_results/output_files",
        "output_files",
        zoning_model_task.output
        ).add_pvolumes({"/mnt": vop})
```
Running the pipeline initialization:
```python
if __name__ == '__main__':
    file_name = "full_belanovica_pipeline.yaml"
    kfp.compiler.Compiler().compile(full_pipeline, file_name)

```

Uploading custom pipeline to Kubeflow Pipelines section by running cell:
```python
import kfp
import random
client=kfp.Client()
filepath = '/home/jovyan/delineation-and-zoning-datavol-1/' + file_name
name = 'ZoningPipeline_timestamp'
print("Uploaded pipeline:", name)
pipeline = client.pipeline_uploads.upload_pipeline(filepath, name=name)
```
New pipeline will be created in Kubflow pipelines section and by creating new run as explained in [here](https://github.com/FlexiGroBots-H2020/AI-platform/blob/c07ef85224c4533fd04f80b07a5ba4398e17597c/kubeflow/Documentation.md#3-pipeline-generation)


As a result of this pipeline, the final zones and sampling points are uploaded to minIO. These files can be downloaded and loaded to QGIS:

![Zones and sampling points](https://github.com/Dimitrije2507/BlueberryRowDetectionKubeflow/blob/c9a1202dc079614375b7e56ef2ba2ece3e24e8d2/results_zones_and_sp.png)
