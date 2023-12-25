# Blueberry row detection - Training a segmentation models with custom data

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

