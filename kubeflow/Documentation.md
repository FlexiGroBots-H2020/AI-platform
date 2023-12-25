# Kubeflow AI Platform - Documentation

## Running a custom ML experiment/Pipeline

Kubeflow Pipelines is a platform for building and deploying portable, scalable machine learning (ML) workflows based on Docker containers

### Preparation steps
In order to run an experiment, there is a preparation procedure that includes following steps

### Pipeline generation

A pipeline is a description of an ML workflow, including all of the components in the workflow and how they combine in the form of a graph. The pipeline includes the definition of the inputs (parameters) required to run the pipeline and the inputs and outputs of each component.
A pipeline component is a self-contained set of user code, packaged as a Docker image, that performs one step in the pipeline. For example, a component can be responsible for data preprocessing, data transformation, model training, and so on. In our typical pipeline configuration there are three components that will be explained below.

### 1. Dockerization - Kubeflow must be provided a docker image of the source code it is running.

__IMPORTANT NOTE__: Including Parser in source code

In order for the Pipeline to be able to run the provided docker image with the appropriate hyperparameters and arguments, the hyperparameters must be parsed by the argparse library.
```python
# An example
parser = argparse.ArgumentParser(description='My program description')
parser.add_argument('--learning_rate', type=str, default="[1e-1]")
parser.add_argument('--lambda_parametar', type=str, default="[2]")
parser.add_argument('--stepovi_arr', type=str, default="[5]")
parser.add_argument('--num_epochs', type=str, default="[1]")
parser.add_argument('--loss_type', type=str, default="['bce']")
parser.add_argument('--Batch_size', type=str, default="[8]")
parser.add_argument('--net_type', type=str, default="SegNet")
parser.add_argument('--device', type=str, default="cpu")
parser.add_argument('--trening_location', type=str, default="trening_set_mini2/img")
parser.add_argument('--validation_location', type=str, default="validation_set_mini2/img")
parser.add_argument('--test_location', type=str, default="test_set_mini2/img")
parser.add_argument('--new_location', type=str)
parser.add_argument('--output1-path', type=str, help='Path of the local file where the Output 1 data should be written.')
args = parser.parse_args()
```

Then, a docker image of each component should be built and it should contain all the scripts, metadata files, json, csv files needed for running the source code.  
Components are divided into a separate folders and by positioning in gitbash in specific component folder we can create docker image of that component. 
When creating a Docker image, you need to create a Dockerfile, which is a script that contains a set of instructions for building a Docker image. The Dockerfile specifies the base image, sets up the environment, copies files into the image, and defines the commands to run when the container starts.

Docker file example:
```python
FROM python:3.8
WORKDIR /pipelines
COPY requirements.txt /pipelines

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y


RUN pip install -r requirements.txt
COPY . /pipelines

RUN chmod +x /pipelines/Train_BGFG_BCE_with_weightsUnet3.py

ENTRYPOINT ["python"]
CMD ["/pipelines/Train_BGFG_BCE_with_weightsUnet3.py"]

```

Creating docker images involves the following steps/commands:

1. Docker build
      The docker build command is used to build a Docker image from a specified Dockerfile that contains commands for running the component in a proper way as well as setting the component environment. Here is the basic syntax:
      ```python
      docker build [OPTIONS] PATH | URL | -
      ```
      Example for building a test component:
      ```python
      docker build -t rebuild_test_component .
      ```

2. Docker tag
      The docker tag command is used to assign a tag to an image. Tags provide a way to give a meaningful and human-readable name to a specific version or variant of an image. Tags are often used to version container images or differentiate between different configurations of the same application.
      ```python
      docker tag SOURCE_IMAGE[:TAG] TARGET_IMAGE[:TAG]
      ```
      Example:
      ```python
      docker tag rebuild_test_component ghcr.io/flexigrobots-h2020/rebuild_test_component
      ```

3. Docker push
      The docker push command is used to push a Docker image or a set of images to a container registry, making them available for others to pull and use. The basic syntax for the docker push command is as follows:
      ```python
      docker push [OPTIONS] NAME[:TAG]
      ```
      Example:
      ```python
      docker push ghcr.io/flexigrobots-h2020/rebuild_test_component
      ```

After these steps, in the our Project Github packages section, an inastance will be created with the previously assigned tag.
![Example_image](https://github.com/Dimitrije2507/BlueberryRowDetectionKubeflow/blob/6883f02980ba7cb51a5e4a58b09173587db4111f/package%20test%20rebuild.png)
It is very important to change package visibility after its initializaion. By default this parameter is set on private, but in order for package/image to be available/visible for our Kubeflow pipelines it should be changed to public. After saving the changes, building and pushing all the necesary images, everything is ready for creating our custom pipelines within the Kubeflow.

### 2. Creating Volume
In Kubelow, you can use volumes to persist and share data between different components of your applications. Volumes in Kubernetes are used to store data independently of the containers running in the pods

![Example_image](https://github.com/Dimitrije2507/BlueberryRowDetectionKubeflow/blob/ced1f14b3d82aca518c6043be87f71e4f5eee241/volumes%20all.png)

Creating New Volume

![Example_image](https://github.com/Dimitrije2507/BlueberryRowDetectionKubeflow/blob/ced1f14b3d82aca518c6043be87f71e4f5eee241/volume%20image.png)

The user should specify name and size, and the rest should be set by default.


### 3. Pipeline generation:

Within this section we will first address creating creating the Notebook where the pipelines will be configured.

![Example_images](https://github.com/Dimitrije2507/BlueberryRowDetectionKubeflow/blob/663d07839b3e2c8b432147fee522b624c4a81ba4/notebooks%20all.png)


Set Notebook name and choose the JupyterLab environment. The name must be in lower cases and words should be separated with dashes.  

![Example_image](https://github.com/Dimitrije2507/BlueberryRowDetectionKubeflow/blob/663d07839b3e2c8b432147fee522b624c4a81ba4/new%20notebook%201.png)

Specify Workplace Volume - Volume that will be mounted in your home directory.

It is possible to attach already existing Volume or to create new one
If the new Volume is created, there will be automatically created folder where all the components of pipeline should be stored/uploaded

Specify Data Volume - Volume that will be mounted in your home directory.

It is possible to attach already existing Volume or to create new one

![Example_image](https://github.com/Dimitrije2507/BlueberryRowDetectionKubeflow/blob/663d07839b3e2c8b432147fee522b624c4a81ba4/new%20notebook%202.png)


After that the notebook is created and the user can connect to it. In the notebook, the user should upload .yaml files for each component of the pipeline and one .ipynb file that will connect them all.

Example

Components: 
* download_model.yaml
* train_model.yaml
* postprocessing_model.yaml

Connecting notebook:
* FullPipeline.ipynb - connects each component and their mutual outputs and inputs (Output of download component with train components input.

To create .yaml file of the whole pipeline run the notebook cells:
```python
import kfp
import kfp.dsl as dsl
from kfp import components
import kubernetes as k8s
from kfp import compiler, dsl
from kubernetes.client import V1VolumeMount
import pprint
import numpy as np


@dsl.pipeline(
    name="blueberry detection",
    description="blueberry detection pipeline"
)
def full_pipeline(size: str="2Gi"):
    
    vop = dsl.PipelineVolume(
        pvc="biosens-volume-main", # change with the corresponding volume name #
        name="biosens-volume-main",
    )

    download_component = kfp.components.load_component_from_file('download_model.yaml')
    train_model = kfp.components.load_component_from_file('train_model.yaml')
    postprocessing_model = kfp.components.load_component_from_file('postprocessing_model.yaml')

```
 
It’s important to specify the name of a volume we are attached to our pipeline, as well as to initialize each component from their respective .yaml file.

Download component initialization
```python
    download_component_task = download_component(
        "blueberry-full", # minIO bucket where the data is storade
        "/mnt"
        ).add_pvolumes({"/mnt": vop})
    download_component_task.execution_options.caching_strategy.max_cache_staleness = "P0D" # avoiding caching
```

Train component initialization

```python
    train_model_task = train_model(
        [1e-3], #'Learning rate'
        [1], #'Lambda parametar'
        [5], #'Stepovi arr'
        [8], #'num_epochs'
        ['bce'], #'Lloss_type'
        [2], #'Batch_size'
        "UNet++", #'Type of Network that will be run. Available model architectures: UNet3, UNet++, SegNet, PSPNet, UperNet, DUC_HDCNet'
        "cuda", #'will torch run on cuda (gpu) or cpu'
        "/mnt/FullSet/trening_set/img/", # 'trening_location'
        "/mnt/FullSet/validation_set/img/", #'validation_location'
        "/mnt/FullSet/test_set/img/", #'test_location'
        # "/mnt" # if download is pruned this should replace the line bellow
        download_component_task.output
        ).add_pvolumes({"/mnt": vop})
```

Limiting number of device that could be used, cache staleness option

Postprocessing component initialization

```python
    train_model_task.set_gpu_limit(1)
    train_model_task.execution_options.caching_strategy.max_cache_staleness = "P0D" # avoiding caching
    postprocessing_model = postprocessing_model(
        "/mnt/logs/Train_BGFG_BCE_with_weightsUNet++/", # path where the logs will be stored during the training
        "blueberry-results", # output minIO bucket for storing results
        train_model_task.output
        ).add_pvolumes({"/mnt": vop})
```

Running the pipeline initialization:
```python
if __name__ == '__main__':
    file_name = "full_nettype_pipeline.yaml" # nettype can be changed with corresponding net type in the moment of creating pipelines
    kfp.compiler.Compiler().compile(full_pipeline, file_name)
```


Uploading custom pipeline to Kubeflow Pipelines section by running cell:

```python
import kfp
import random
client=kfp.Client()
filepath = '/home/jovyan/biosens-volume-main/' + file_name
name = 'train_pipeline_nettype_timestamp' # nettype and timestamp can be changed with corresponding net type and time in the moment of creating pipelines
print("Uploaded pipeline:", name)
pipeline = client.pipeline_uploads.upload_pipeline(filepath, name=name)
```

New pipeline will be created:

![Example_image](https://github.com/Dimitrije2507/BlueberryRowDetectionKubeflow/blob/392b98cd8b31544cb182f287b6b174ac422e46f2/new%20notebook%203.png)

To create new pipeline run, experiment must be specified and previously created

![Example_image](https://github.com/Dimitrije2507/BlueberryRowDetectionKubeflow/blob/ced5d9e74fdc400857c3f69300c2528771945303/runs%201.png)

Specify name and description of the experiment
In the last 5 runs section there is a preview of last 5 runs statuses:

![Example_images](https://github.com/Dimitrije2507/BlueberryRowDetectionKubeflow/blob/c19e03585d997853d4edface6ed62ff38f4520b8/statuses.png)

Pipeline structure is shown in the figure below. 

![Example_image](https://github.com/Dimitrije2507/BlueberryRowDetectionKubeflow/blob/bd5147d6db574cf339823b4e5370437ad08fef67/runs%202.png)


Download component - used for downloading data from data storage (in our case from minIO bucket) to the local Kubeflow volume.

Main component - used for executing learning procedures, training, testing scripts, inference services or any pipeline that is supposed to run on the AI platform.

Push component - used for uploading results to the data storage (also minIO in our case).

Additional components - besides these three basic pipeline components, there could be additional components or branches of pipeline that could be used for data visualization, results evaluation, explainable AI or simply for different types of data processing etc.

![Example_image](https://github.com/Dimitrije2507/BlueberryRowDetectionKubeflow/blob/137469209187ed51160448666c2d8057ffcfd58c/runs%203.png)

Choose a previously generated experiment and click “use this experiment”. Then you can start the run

![Example_image](https://github.com/Dimitrije2507/BlueberryRowDetectionKubeflow/blob/1b65ab920023698d568256b8ac495259c661721c/runs%204.png)

New run will be shown in Runs section

![Example_image](https://github.com/Dimitrije2507/BlueberryRowDetectionKubeflow/blob/0b7b6050f016b6adbb389f67283f7fde43463ec4/runs%205.png)









 ## Blueberry row detection - Training a segmentation models with custom data

[Steps for running blueberry row detection model training](https://github.com/FlexiGroBots-H2020/AI-platform/blob/ee8190b157c13eb15890ad94194a6b0b401d7a87/kubeflow/Blueberry_row_detection/Training/BBRTReadMe.md)


## Running an AutoML experiment

Documentation and guidelines for running Katib/AutoML experiment, as well as the examples are provided [here](https://github.com/FlexiGroBots-H2020/AI-platform/tree/ee8190b157c13eb15890ad94194a6b0b401d7a87/kubeflow/Katib)

Another example of running AutoML hyperparameter search for training blueberry rows detection model is explained [here](https://github.com/FlexiGroBots-H2020/AI-platform/blob/35e3dcbf5d991c1461a63dec22a265ea2654b356/kubeflow/Katib/docker%20images/Katib-BBRowDetection/README.md). All the scripts needed for reproducing this experiment can also be found on the link above.



## Blueberry row detection - Inference service

We developed an inference service pipeline for blueberry row detection using state-of-the-art deep learning models (trained using AI platform).  
[Steps for running blueberry row detection inference service (model testing)](https://github.com/FlexiGroBots-H2020/AI-platform/blob/3a29eadbecf608826585986b9f3709f9687aa49f/kubeflow/Blueberry_row_detection/Test/BBRInferenceServiceReadMe.md).


## Vegetation index maps calculator - Inference service

We developed an inference service pipeline for calculating vegetation index(VI) maps using UAV multispectral data/orthomosaics. 
[Steps for running vegetation index maps calculator](https://github.com/FlexiGroBots-H2020/AI-platform/blob/90a38cff49f61f13e3e888f395406fea9fcf6ad0/kubeflow/Vegetation%20indices%20calc/VegetationCalculationReadMe.md)

## Delineation and zoning - Inference service

We developed an inference service pipeline for Delineation and zoning using UAV multispectral data/orthomosaics and ML techniques.
[Steps for running delineation and zoning inference service](https://github.com/FlexiGroBots-H2020/AI-platform/blob/42fb7de031a4124d02d264b522570384deca061d/kubeflow/Delineation%20Inference/DelineationAndZoningReadMe.md) 





