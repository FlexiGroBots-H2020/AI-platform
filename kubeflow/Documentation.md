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

### Steps for running blueberry row detection model training:

We developed a pipeline for training blueberry row detection using several state-of-the-art deep learning models. The pipeline consists of three components or blocks. Through this pipeline we can firstly download data of interest from the data storage (in our case it’s minIO), then arrange a desired experiment through adjusting hyperparameters such as batch size, number of epochs, learning rate etc and lastly provide the trained model back to our data storage including the training metrics. 

![Example Image](https://github.com/Dimitrije2507/BlueberryRowDetectionKubeflow/blob/main/pipeline_example.png)

#### Step 1: Creating notebook
![Example Image](https://github.com/Dimitrije2507/BlueberryRowDetectionKubeflow/blob/33e6b02673eb42f81c85781c0169b00b14bf246d/notebooks.png)
#### Step 2: Uploading yaml files for each component

YAML files for this and every other pipeline we developed can be found in FlexiGroBots [Github](https://github.com/FlexiGroBots-H2020/AI-platform/tree/biosens/kubeflow/Blueberry_row_detection/Training) repo in the corresponding folders under the biosens branch. Each file assigns appropriate docker image/package:
- Download component: training_download_component 
- Training component: blueberry_rebuild_test1
- PostProcessing component: rebuild_postprocessing_component.

These docker images were pushed to the project GitHub repo and can be found in [packages](https://github.com/orgs/FlexiGroBots-H2020/packages) section.

![Example Image](https://github.com/Dimitrije2507/BlueberryRowDetectionKubeflow/blob/6701f1114db73a3cd8eacb597b8d932bd690e50d/packages.png)

An IPYNB file is also necessary and it is included in the same folders. This file is essential for connecting individual components. It consists of at least two major cells. In the first one it’s always important to check volume name in the code and change it to match with corresponding volume:

![Example Image](https://github.com/Dimitrije2507/BlueberryRowDetectionKubeflow/blob/6701f1114db73a3cd8eacb597b8d932bd690e50d/volumes_names.png)

Loading components from files:

```python
download_component = kfp.components.load_component_from_file('download_model.yaml')
train_model = kfp.components.load_component_from_file('train_model.yaml')
postprocessing_model = kfp.components.load_component_from_file('postprocessing_model.yaml')
```
Initializing components and connecting them through the arguments of their functions:

```python
download_component_task = download_component(
      "blueberry-full", # minIO bucket where the data is storade
      "/mnt"
      ).add_pvolumes({"/mnt": vop})
download_component_task.execution_options.caching_strategy.max_cache_staleness = "P0D" # avoiding caching
train_model_task = train_model(
      [1e-3], #'Learning rate'
      [1], #'Lambda parametar'
      [5], #'Stepovi arr'
      [8], #'num_epochs'
      ['bce'], #'Lloss_type'
      [2], #'Batch_size'
      "DUC_HDCNet", #'Type of Network that will be run. Available model architectures: UNet3, UNet++, SegNet, PSPNet, UperNet, DUC_HDCNet'
      "cuda", #'will torch run on cuda (gpu) or cpu'
      "/mnt/FullSet/trening_set/img/", # 'trening_location'
      "/mnt/FullSet/validation_set/img/", #'validation_location'
      "/mnt/FullSet/test_set/img/", #'test_location'
      # "/mnt" # if download is pruned this should replace the line bellow
      download_component_task.output
      ).add_pvolumes({"/mnt": vop})

train_model_task.set_gpu_limit(1)
train_model_task.execution_options.caching_strategy.max_cache_staleness = "P0D" # avoiding caching
postprocessing_model = postprocessing_model(
      "/mnt/logs/Train_BGFG_BCE_with_weightsDUC_HDCNet/",
      "blueberry-results",
      train_model_task.output
      ).add_pvolumes({"/mnt": vop})
```
Compiling and creating the final YAML file that accompanies all the components:

```python
if __name__ == '__main__':
    file_name = "full_net_type_pipeline.yaml"
    kfp.compiler.Compiler().compile(full_pipeline, file_name)
```
In the second cell the corresponding pipeline is created in the Kubeflow platform.
```python
import kfp
import random
client=kfp.Client()
filepath = '/home/jovyan/biosens-volume-main/' + file_name
name = 'train_pipeline_net_type_time_stamp'
print("Uploaded pipeline:", name)
pipeline = client.pipeline_uploads.upload_pipeline(filepath, name=name)
```
Back in the Kubeflow, under the pipelines section there should be a new pipeline with the name we specified in the notebook.
Running the experiment should be the same as mentioned in “Running a custom experiment” section.

__Note__:
After running the pipeline, the download component will download the data to the local volume and the data will be accessible until it is deleted. After running the pipeline again, data will be overwritten, but nothing will change as it is the same data. If there is a need for frequent running of the same pipeline (for example because of changing some hyperparameter), to avoid waiting for the download component to be executed, we advise to prune the download component.

## Running an AutoML experiment

Documentation and guidelines for running Katib/AutoML experiment, as well as the examples are provided [here](https://github.com/FlexiGroBots-H2020/AI-platform/tree/biosens/kubeflow/Katib)

Another example of running AutoML hyperparameter search for training blueberry rows detection model will be explained below. All the scripts needed for reproducing this experiment can also be found on the link above.

As explained in the Katib doc, AutoML experiments can be created through the Kubeflow SDK in the AutoML Experiments section, but it is also possible to generate these experiments through the notebook. This approach allows us to generate YAML files in a more intuitive way then through the SDK as it gives us insight whats happening under the hub, while in SDK these YAML files are meant to be written on our own (or copied from the existing files) which could lead to unintended errors.

We will use [Katib-BBRowDetection.ipynb](https://github.com/FlexiGroBots-H2020/AI-platform/blob/biosens/kubeflow/Katib/ipynb%20files/Katib-BBRowDetection.ipynb) as an example notebook. 

### Initial steps
Initial steps include installing:
```python
!pip install kfp==1.8.18
!pip install kubeflow-katib==0.14.0
```
as well as importing corresponding Kubeflow libs:
```python
import kfp
import kfp.dsl as dsl
from kfp import components
from kubeflow.katib import KatibClient
from kubeflow.katib import ApiClient
from kubeflow.katib import V1beta1ExperimentSpec
from kubeflow.katib import V1beta1AlgorithmSpec
from kubeflow.katib import V1beta1ObjectiveSpec
from kubeflow.katib import V1beta1ParameterSpec
from kubeflow.katib import V1beta1FeasibleSpace
from kubeflow.katib import V1beta1TrialTemplate
from kubeflow.katib import V1beta1TrialParameterSpec
from kubeflow.katib import V1beta1Experiment
from kubernetes.client import V1ObjectMeta
from kubeflow.katib import V1beta1MetricsCollectorSpec
from kubeflow.katib import V1beta1SourceSpec
from kubeflow.katib import V1beta1FileSystemPath
from kubeflow.katib import V1beta1CollectorSpec
from kubeflow.katib import V1beta1FilterSpec
```
The next steps represent experiment configuration in the same way as in SDK:

### 1. Metadata - defining experiment name and namespace
```python
experiment_name = "katib-bbrow-detection-rebuild-final"
namespace = "dimitrijestefanovic97"

metadata = V1ObjectMeta(
    name=experiment_name,
    namespace=namespace
)
```
### 2. Trial threshold - # Specifying trial thresholds such as maximum number of trial, maximum number of failed trials or number of trials that run in parallel

```python
max_trial_count = 20
max_failed_trial_count = 5
parallel_trial_count = 3
```
### 3. Objective - Specifying what is the metric we will monitor during parameter tuning as well as the goal.

```python
objective = V1beta1ObjectiveSpec(
    type="maximize",
    goal=0.99,
    objective_metric_name="accuracy"
)
```
### 4. Search Algorithm - The Search Algorithm is responsible for navigating through the optimization search space
```python
algorithm = V1beta1AlgorithmSpec(
    algorithm_name="grid",
)
```
### 5. Early stopping - Allows you to avoid overfitting when you train your model during Katib Experiments. In our case it was skiped
### 6. Hyperparameters - These will be used to construct the optimization search space. Katib will be generating Trials to test different combinations of these parameters in order to find the optimal set
```python
# In this example we tune learning rate, batch size, number of epochs and architecture type.
parameters = [
    V1beta1ParameterSpec(
        name="learning_rate",
        parameter_type="double",
        feasible_space=V1beta1FeasibleSpace(
            min="0.001",
            max="0.003",
            step = "0.001"
        ),
    ),
    V1beta1ParameterSpec(
      name="net_type",
      parameter_type="categorical",
      feasible_space=V1beta1FeasibleSpace(
        list=["UNet3", "UNet++","SegNet"]#,"DUC_HDCNet"]
      ),
    ),
    V1beta1ParameterSpec(
        name="num_epochs",
        parameter_type="int",
        feasible_space=V1beta1FeasibleSpace(
            min="4",
            max="5",
            step = "1"
        ),
    ),
    V1beta1ParameterSpec(
        name="Batch_size",
        parameter_type="int",
        feasible_space=V1beta1FeasibleSpace(
            min="2",
            max="4",
            step = "2"
        ),
    )
]
```
### 7. Metrics Collector - Define how Katib should collect the metrics from
```python
metrics = V1beta1MetricsCollectorSpec(
    source=V1beta1SourceSpec(
        filter=V1beta1FilterSpec(
            metrics_format=["{metricName: ([\\w|-]+), metricValue: ((-?\\d+)(\\.\\d+)?)}"]
        ),
        file_system_path=V1beta1FileSystemPath(
            path="/katib/mnist.log",
            kind="File"
        )
    ),
    collector=V1beta1CollectorSpec(
        kind="File"
    )
)

```
### 8. Experimental trial template - Define the Trial's YAML.
```python
trial_spec = {
    "apiVersion": "batch/v1",
    "kind": "Job",
    "spec": {
        "template": {
            "metadata": {
                "annotations": {
                    "sidecar.istio.io/inject": "false"
                }
            },
            "spec": { 
                "containers": [
                            {
                    "name": "training-container",
                    "image": "ghcr.io/flexigrobots-h2020/pipelinetesting3:latest",
                    "command": [
                        "python3",
                         "Train_BGFG_BCE_with_weightsUnet3.py",
                        "--learning_rate=${trialParameters.learningRate}",
                        "--num_epochs=${trialParameters.numEpochs}",
                        "--Batch_size=${trialParameters.batchSize}",
                        "--net_type=${trialParameters.netType}",
                        "--trening_location=trening_set_mini2/img",
                        "--validation_location=validation_set_mini2/img",
                        "--test_location=test_set_mini2/img",
                        "--new_location=/katib/mnist.log"
                                ],
                     }
                        ],
                "restartPolicy": "Never",
            }
        }
    }
}
```
__Important note__: Be sure to change the “image” parameter to corresponding docker image. This image should have an appropriate deep learning pipeline for which we are trying to find optimal hyper parameters (in this case blueberry row detection).
Another important substep is to configure parameters for trial template, that is to connect trial parameter instances and defined hyperparameters:

```python
# Configure parameters for the Trial template.
trial_template = V1beta1TrialTemplate(
    primary_container_name="training-container",
    trial_parameters=[
        V1beta1TrialParameterSpec(
            name="learningRate",
            description="Learning rate for the training model",
            reference="learning_rate"
        ),
        V1beta1TrialParameterSpec(
            name="numEpochs",
            description="number of epochs for training",
            reference="num_epochs"
        ),
        V1beta1TrialParameterSpec(
            name="batchSize",
            description="number of samples in one batch",
            reference="Batch_size"
        ),
        V1beta1TrialParameterSpec(
            name="netType",
            description="type of model",
            reference="net_type"
        )
    ],
    trial_spec=trial_spec
)
```

### Generating an experiment
Now that we defined all the instances and components, everything is ready for generating an experiment:


```python

# Create an Experiment from the above parameters.
experiment_spec = V1beta1ExperimentSpec(
    max_trial_count=max_trial_count,
    max_failed_trial_count=max_failed_trial_count,
    parallel_trial_count=parallel_trial_count,
    objective=objective,
    algorithm=algorithm,
    parameters=parameters,
    trial_template=trial_template,
    metrics_collector_spec=metrics
)

experiment = V1beta1Experiment(
    api_version="kubeflow.org/v1beta1",
    kind="Experiment",
    metadata=metadata,
    spec=experiment_spec
)
```

The final step includes creating Katib client and corresponding experiment in specified Kubeflow namespace  

```python
# Create client.
kclient = KatibClient()

# Create your Experiment.
kclient.create_experiment(experiment,namespace="dimitrijestefanovic97")
```

Underneath the final cell in the notebook, there should be the experiment configuration printed after its execution. Also there should be a link that leads to the Kubeflow SDK and its AutoML experiments section.

![Example_image](https://github.com/Dimitrije2507/BlueberryRowDetectionKubeflow/blob/afeecb8bc5b08c2375eb2657ec423d0d61094a36/experiments2.png)

### Experiment details:

![Example_image](https://github.com/Dimitrije2507/BlueberryRowDetectionKubeflow/blob/f968fa7b05d4004333efc31eee2bd0a35d689374/details.png)

Kubeflow SDK gives us live overview of AutoML experiments and what combination of hyperparameters is potentially the optimal one with corresponding metric such as accuracy:

![Example_image](https://github.com/Dimitrije2507/BlueberryRowDetectionKubeflow/blob/97054d0011b020a2c38a049179abb26d9becfee3/overview.png)

In this experiment the hyperparameters we are trying to tune are learning rate, batch size and net type, that is we are also trying to figure out which segmentation model architecture is the most suitable for this task.

Each trial can be observed separately and inspected. It should be noted that as soon there is new potentially optimal trial it is highlighted in yellow:

![Example_image](https://github.com/Dimitrije2507/BlueberryRowDetectionKubeflow/blob/0d5485dcd5961bdcd20ff7032dbe6506c6371f28/optimal_yellow.png)

For individual trials there are logs that can be tracked in real time: 

![Example_image](https://github.com/Dimitrije2507/BlueberryRowDetectionKubeflow/blob/9684221f191801dedfac13838faf49c05ee96aa8/logs.png)

After all the trials are finished, the combination of hyper parameters with the best metrics can be used in the classic training procedure explained in the previous section and it should provide reliable segmentation model.


## Blueberry row detection - Inference service

We developed an inference service pipeline for blueberry row detection using state-of-the-art deep learning models (trained using AI platform). The pipeline consists of three components or blocks. Through this pipeline we must firstly download data from the data storage (minIO), then we have to choose a part of parcel where we want to detect blueberry rows and lastly provide the detected rows back to our data storage in tif, shp and png formats, including the metrics that describe how accurate the model was. 

#### Step 1: Creating notebook
#### Step 2: Uploading yaml files for each component

YAML files for this pipeline can be found in FlexiGroBots [Github](https://github.com/FlexiGroBots-H2020/AI-platform/tree/biosens/kubeflow/Blueberry_row_detection/Test) repo in the corresponding folders under the biosens branch. Each file assigns appropriate docker image/package:
- Download component: training_download_component 
- Test component: rebuild_test_component
- PostProcessing component: rebuild_test_push_component2.

These docker images were pushed to the project GitHub repo and can be found in [packages](https://github.com/orgs/FlexiGroBots-H2020/packages) section.

An IPYNB file is also necessary and it is included in the same folders in GitHub repo. This file is essential for connecting individual components. It consists of at least two major cells. In the first one it’s always important to check volume name in the code and change it to match with corresponding volume:

The rest is the same as for Blueberry row detection training pipeline.


## Vegetation index maps calculator - Inference service

We developed an inference service pipeline for calculating vegetation index(VI) maps using UAV multispectral data/orthomosaics. The pipeline consists of three components or blocks. Through this pipeline, as in all previous pipelines, we must firstly download data from the data storage (minIO), then we have to choose a part of parcel where we want to calculate VI maps and lastly provide the maps back to our data storage in tif or any other suitable format.

#### Step 1: Creating notebook
#### Step 2: Uploading yaml files for each component

YAML files for this pipeline can be found in FlexiGroBots [Github](https://github.com/FlexiGroBots-H2020/AI-platform/tree/biosens/kubeflow/Vegetation%20Indices) repo in the corresponding folders under the biosens branch. Each file assigns appropriate docker image/package:
- Download component: rebuild_delineation_download_component 
- Test component: vegetation_ind_calc_component
- PostProcessing component: rebuild_delineation_postprocessing_component.

These docker images were pushed to the project GitHub repo and can be found in [packages](https://github.com/orgs/FlexiGroBots-H2020/packages) section.

An IPYNB file is also necessary and it is included in the same folders in GitHub repo. This file is essential for connecting individual components. It consists of at least two major cells. In the first one it’s always important to check volume name in the code and change it to match with corresponding volume:

The rest is the same as for Blueberry row detection training and test pipelines.


## Delineation and zoning - Inference service

We developed an inference service pipeline for Delineation and zoning using UAV multispectral data/orthomosaics and ML techniques. The pipeline consists of three components or blocks. Again, we must first download data from the data storage (minIO), then we have to choose a part of parcel where we want to calculate vegetation indices and generate maps that are further being used in ML pipeline for delineation and zoning. Lastly we need to provide the parcel zones back to our data storage in tif, shp or any other suitable format.

#### Step 1: Creating notebook
#### Step 2: Uploading yaml files for each component

YAML files for this pipeline can be found in FlexiGroBots [Github](https://github.com/FlexiGroBots-H2020/AI-platform/tree/biosens/kubeflow/Delineation%20Inference) repo in the corresponding folders under the biosens branch. Each file assigns appropriate docker image/package:
- Download component: rebuild_delineation_download_component 
- Test component: rebuild_delineation_test_component
- PostProcessing component: rebuild_delineation_postprocessing_component.

These docker images were pushed to the project GitHub repo and can be found in [packages](https://github.com/orgs/FlexiGroBots-H2020/packages) section.

An IPYNB file is also necessary and it is included in the same folders in GitHub repo. This file is essential for connecting individual components. It consists of at least two major cells. In the first one it’s always important to check volume name in the code and change it to match with corresponding volume:

The rest is the same as for Blueberry row detection training and test pipelines.

## Delineation and zoning + Soil Sampling point generation



