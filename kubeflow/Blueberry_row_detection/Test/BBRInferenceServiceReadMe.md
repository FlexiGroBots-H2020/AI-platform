# Blueberry row detection - Inference service

The pipeline consists of three components or blocks. Through this pipeline we must firstly download data from the data storage (minIO), then we have to choose a part of parcel where we want to detect blueberry rows and lastly provide the detected rows back to our data storage in tif, shp and png formats, including the metrics that describe how accurate the model was. 

#### Step 1: Creating notebook

The user can create new notebook for this task, as explained in [here](https://github.com/FlexiGroBots-H2020/AI-platform/blob/c07ef85224c4533fd04f80b07a5ba4398e17597c/kubeflow/Documentation.md#3-pipeline-generation), or use already existing.

#### Step 2: Uploading yaml files for each component

YAML files for this pipeline as well as all other necessary files can be found in FlexiGroBots [Github](https://github.com/FlexiGroBots-H2020/AI-platform/tree/a9c0131b69ea059fb40281bbac761ddd8ae81a36/kubeflow/Blueberry_row_detection/Test) repo in the corresponding folders. Each file assigns appropriate docker image/package:
- Download component: training_download_component 
- Test component: rebuild_test_component
- PostProcessing component: rebuild_test_push_component2.

These docker images were pushed to the project GitHub repo and can be found in [packages](https://github.com/orgs/FlexiGroBots-H2020/packages) section.

An IPYNB file is also necessary and it is included in the same folders in GitHub repo. This file is essential for connecting individual components. It consists of at least two major cells. In the first one it’s always important to check volume name in the code and change it to match with corresponding volume:

![Volume_name_image](https://github.com/Dimitrije2507/BlueberryRowDetectionKubeflow/blob/92e24d80b3c29ae9bbdfc0092a8706a82a7d1525/namevol.png)

After that, the user should initialize each pipeline component by loading them from corresponding yaml files: 

```python
download_component = kfp.components.load_component_from_file('download_model.yaml')
test_component = kfp.components.load_component_from_file('test_component.yaml')
postprocessing_model = kfp.components.load_component_from_file('postprocessing_model.yaml')
```

Specifying Blueberry row detection model path :

```python
model_dir = "/mnt/model/"
model_name = "blueberry_row_detection_model.pt"
model_path = model_dir + model_name
```
This model can be uploaded manually to the notebook or downloaded from the minIO through the download component. In both cases it should be placed in model folder within the notebook

Initialization of download component:
```python
download_component_task = download_component(
    "biosens-test-inference-service", # minIO bucket where the data is stored
    "/mnt" # root folder in the notebook
    ).add_pvolumes({"/mnt": vop})
download_component_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
```

Initialization of Blueberry row detection - test component:

```python
test_component_task = test_component(
    model_path,
    "UnetPlusPlus", # specifying model type
    "cuda", # specifying the device for running inference service
    download_component_task.output
    ).add_pvolumes({"/mnt": vop})
test_component_task.set_gpu_limit(1)
test_component_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
```

Initialization of postprocessing component used for uploading results to minIO:

```python
postprocessing_model = postprocessing_model(
    "blueberry-results", # minIO bucket for uploading results
    "/mnt/DataTest/final_results_folder/unetpp_test_parcel_belanovica.tiff", # minIO path for uploading tif file that contains detected rows  
    "/mnt/DataTest/final_results_folder/unetpp_test_rgb_masked_with_red_belanovica.tiff", # minIO path for uploading tif file that contains detected rows overlaped on rgb orthomosaic 
    "/mnt/DataTest/final_results_folder/unetpp_test_parcel_belanovica.shp", # minIO path for uploading shp file that contains detected rows
    test_component_task.output).add_pvolumes({"/mnt": vop}) 
```

Running the pipeline initialization:

```python

if __name__ == '__main__':
    file_name = "full_pipeline_belanovic1.yaml"
    kfp.compiler.Compiler().compile(full_pipeline, file_name)

```

Uploading custom pipeline to Kubeflow Pipelines section by running cell:

```python

import kfp
import random
client=kfp.Client()
filepath = '/home/jovyan/test-notebook-datavol-1/' + file_name
name = 'Blueberry_test_timestamp'
print("Uploaded pipeline:", name)
pipeline = client.pipeline_uploads.upload_pipeline(filepath, name=name)

```

New pipeline will be created in Kubflow pipelines section and by creating new run as explained in [here](https://github.com/FlexiGroBots-H2020/AI-platform/blob/c07ef85224c4533fd04f80b07a5ba4398e17597c/kubeflow/Documentation.md#3-pipeline-generation)

Resulting tif maps and shp file stored in the minIO: 

![Results_minIO_image](https://github.com/Dimitrije2507/BlueberryRowDetectionKubeflow/blob/1fdf27777904951a9967c367f8ffb9a5ab88aad2/resultsmnio.png)

These results can than be downloaded and loaded in QGIS:

![Detected_rows](https://github.com/Dimitrije2507/BlueberryRowDetectionKubeflow/blob/7bf19c1c7aeb470de78475c006a7880b2aaa0499/detected_rows.png)

