# Blueberry row detection - Inference service

The pipeline consists of three components or blocks. Through this pipeline we must firstly download data from the data storage (minIO), then we have to choose a part of parcel where we want to detect blueberry rows and lastly provide the detected rows back to our data storage in tif, shp and png formats, including the metrics that describe how accurate the model was. 

#### Step 1: Creating notebook
#### Step 2: Uploading yaml files for each component

YAML files for this pipeline can be found in FlexiGroBots [Github](https://github.com/FlexiGroBots-H2020/AI-platform/tree/biosens/kubeflow/Blueberry_row_detection/Test) repo in the corresponding folders under the biosens branch. Each file assigns appropriate docker image/package:
- Download component: training_download_component 
- Test component: rebuild_test_component
- PostProcessing component: rebuild_test_push_component2.

These docker images were pushed to the project GitHub repo and can be found in [packages](https://github.com/orgs/FlexiGroBots-H2020/packages) section.

An IPYNB file is also necessary and it is included in the same folders in GitHub repo. This file is essential for connecting individual components. It consists of at least two major cells. In the first one it’s always important to check volume name in the code and change it to match with corresponding volume:

The rest is the same as for Blueberry row detection training pipeline.
