# Running an AutoML experiment


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

