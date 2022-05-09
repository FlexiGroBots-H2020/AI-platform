# Example: Creating pipeline and inference service in Kubeflow

This folder contains all necessary files to reproduce an example where a pipeline and an inference service are deployed.

## Steps to deploy an inference service

### Developing model inference 
The first step to deploy the inference service is to develope the inference of any model in a Python file. In this example, it is found in the [inference script](/kubeflow/example/Inference/inference_service.py).

### Specifying requirements
Before creating a container from the Python file, it is mandatory to freeze the necessary libraries as requirements to install. In [requirements file](/kubeflow/example/Inference/requirements.txt) it can be shown the most important libraries to take into account and its version.

### Dockerizing

#### Dockerfile
Once logic and requirements file are ready, it is the moment to develop a Dockerfile which allows to create a container from the initial Python file and installing all necessary requirements. This [Dockerfile](/kubeflow/example/Inference/Dockerfile) containes all commands to create it.

#### Building and pushing the container
After coding the Dockerfile, it is necessary to build an image and to push it to a remote repository where it can be accessible to use.

Executing from the Dockerfile folder:

```bash
docker build -t <image_name> .
```

```bash
docker tag <image_name> <remote_repo_name>/<image_name>
```

```bash
docker push <remote_repo_name>/<image_name>
```

### Using docker image
The docker image, that has been generated in previous step, will be used in a .yaml file as the container of the desired implementation. This procedure can be found in this [.yaml file](/kubeflow/example/Inference/inference_service.yaml).

### Defining and creating Kubeflow Service
Once the .yaml file is ready, it will be used to define any development as a Kubeflow component. The code that explain this definition process can be found in [the main notebook](/kubeflow/example/demo-notebook.ipynb) of the example, specifically, in the "Define Kubeflow Inference Service and then create service" paragraph.  

### Uploading service
After creating the service, it has to be uploaded to Kubeflow before running. In the paragraph "Upload pipeline" of [the main notebook](/kubeflow/example/demo-notebook.ipynb) can be found this process.

### Running pipeline or service
The last step of a Kubeflow service deployment consists on running the pipeline that has been created after uploading it.

To execute the pipeline/service, users have to go to "Pipelines" menu inside Kubeflow, click on the pipeline that has been uploaded and click on "Create run". In the "Start a run" view, it is mandatory to choose some related experiment to this run.

Finally, going to "Run" menu inside Kubeflow, the status of the service/pipeline deployment can be checked.

