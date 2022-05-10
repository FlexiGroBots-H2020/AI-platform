# Example: Creating pipeline and inference service in Kubeflow

This folder contains all necessary files to reproduce an example where a pipeline and an inference service are deployed.

## Steps to deploy a pipeline

The pipeline described in this document is composed of the following components:

* Downloading data:  to download the required dataset from MinIO.
* Preprocessing data: to split the dataset into training and test before feeding the model.
* Model training: to perform model training and return accuracy.


### Developing the logic of each component 
The first step to deploy a pipeline is to develop the logic of all component that build this pipeline. In this example, the pipeline is composed by three components, so, it is needed to code three diferent Python files for each one. These scripts are: [downloading data](/kubeflow/example/Download_data/download_data.py), [preprocessing data](/kubeflow/example/Preprocess_data/preprocess_data.py) and [model training](/kubeflow/example/Train_model/train_model.py).

### Specifying requirements
Before creating a container from these Python files, it is mandatory to freeze the necessary libraries as requirements to install. In [downloading libraries](/kubeflow/example/Download_data/requirements.txt), [preprocessing libraries](/kubeflow/example/Preprocess_data/requirements.txt) and [training libraries](/kubeflow/example/Train_model/requirements.txt) it can be observed the most important libraries to take into account and its version.

### Dockerizing

#### Dockerfiles
Once logic and requirements files are ready, it is the moment to develop three different, but similar, Dockerfiles which allow to create three containers from the initial Python files, installing all necessary libraries. Next files contains all commands to create the containers: [downloading Dockerfile](/kubeflow/example/Download_data/Dockerfile), [preprocessing Dockerfile](/kubeflow/example/Preprocess_data/Dockerfile) and [training Dockerfile](/kubeflow/example/Train_model/Dockerfile).

#### Building and pushing the container
After coding the Dockerfile, it is necessary to build each image and to push it to a remote repository where it can be accessible to use.

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

### Using docker images
These three docker images, that have been generated in previous steps, will be used in different .yaml files as the containers of the desired implementation. This procedure can be found in: [downloading yaml file](/kubeflow/example/Download_data/download_data.yaml), [preprocessing yaml file](/kubeflow/example/Preprocess_data/preprocess_data.yaml) and [training yaml file](/kubeflow/example/Train_model/train_model.yaml).

### Defining and creating the pipeline
Once all .yaml file are ready, each of them will be used to define each development as a Kubeflow component. The code that explain this definition process can be found in [the main notebook](/kubeflow/example/demo-notebook.ipynb) of the example, specifically, in the "Define Kubeflow Pipeline and then create service" paragraph.  

### Uploading pipeline
After creating the pipeline, it has to be uploaded to Kubeflow before running. In the paragraph "Upload pipeline" of [the main notebook](/kubeflow/example/demo-notebook.ipynb) can be found this process.

### Running pipeline
The last step of a Kubeflow service deployment consists on running the pipeline that has been created after uploading it.

To execute the pipeline, users have to go to "Pipelines" menu inside Kubeflow, click on the pipeline that has been uploaded and click on "Create run". In the "Start a run" view, it is mandatory to choose, or create, some related experiment to this run.

Finally, going to "Run" menu inside Kubeflow, the status of the pipeline deployment can be checked.

