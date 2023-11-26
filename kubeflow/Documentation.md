# Kubeflow AI Platform - Documentation

## Running a custom experiment

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

An IPYNB file is also necessary and it is included in the same folders. As we mentioned in the first section this file is essential for connecting individual components. It consists of at least two major cells. In the first one it’s always important to check volume name in the code and change it to match with corresponding volume:

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

Another example of running AutoML hyperparameter search for training blueberry rows detection model will be explained below.

As explained in the Katib doc, AutoML experiments can be created through the Kubeflow SDK in the AutoML Experiments section, but it is also possible to generate these experiments through the notebook. This approach allows us to generate YAML files in a more intuitive way then through the SDK as it gives us insight whats happening under the hub, while in SDK these YAML files are meant to be written on our own (or copied from the existing files) which could lead to unintended errors.

We will use [Katib-BBRowDetection.ipynb](https://github.com/FlexiGroBots-H2020/AI-platform/blob/biosens/kubeflow/Katib/ipynb%20files/Katib-BBRowDetection.ipynb) as an example notebook. 

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

1. Metadata - defining experiment name and namespace
```python
experiment_name = "katib-bbrow-detection-rebuild-final"
namespace = "dimitrijestefanovic97"

metadata = V1ObjectMeta(
    name=experiment_name,
    namespace=namespace
)
```
2. Trial threshold - # Specifying trial thresholds such as maximum number of trial, maximum number of failed trials or number of trials that run in parallel

```python
max_trial_count = 20
max_failed_trial_count = 5
parallel_trial_count = 3
```
3. Objective - Specifying what is the metric we will monitor during parameter tuning as well as the goal.

```python
objective = V1beta1ObjectiveSpec(
    type="maximize",
    goal=0.99,
    objective_metric_name="accuracy"
)
```
4. Search Algorithm - The Search Algorithm is responsible for navigating through the optimization search space
```python
algorithm = V1beta1AlgorithmSpec(
    algorithm_name="grid",
)
```
5. Early stopping - Allows you to avoid overfitting when you train your model during Katib Experiments. In our case it was skiped
6. Hyperparameters - These will be used to construct the optimization search space. Katib will be generating Trials to test different combinations of these parameters in order to find the optimal set
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
7. Metrics Collector - Define how Katib should collect the metrics from
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
8. Experimental trial template - Define the Trial's YAML.
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
## Blueberry row detection - Inference service

## Delineation and zoning - Inference service

## Delineation and zoning + Soil Sampling point generation



