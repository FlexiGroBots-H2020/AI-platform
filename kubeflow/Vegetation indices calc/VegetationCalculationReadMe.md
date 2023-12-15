# Vegetation index maps calculator - Inference service

The pipeline consists of three components or blocks. Through this pipeline, as in all previous pipelines, we must firstly download data from the data storage (minIO), then we have to choose a part of parcel where we want to calculate VI maps and lastly provide the maps back to our data storage in tif or any other suitable format.

#### Step 1: Creating notebook
#### Step 2: Uploading yaml files for each component

YAML files for this pipeline can be found in FlexiGroBots [Github](https://github.com/FlexiGroBots-H2020/AI-platform/tree/biosens/kubeflow/Vegetation%20Indices) repo in the corresponding folders under the biosens branch. Each file assigns appropriate docker image/package:
- Download component: rebuild_delineation_download_component 
- Test component: vegetation_ind_calc_component
- PostProcessing component: rebuild_delineation_postprocessing_component.

These docker images were pushed to the project GitHub repo and can be found in [packages](https://github.com/orgs/FlexiGroBots-H2020/packages) section.

An IPYNB file is also necessary and it is included in the same folders in GitHub repo. This file is essential for connecting individual components. It consists of at least two major cells. In the first one it’s always important to check volume name in the code and change it to match with corresponding volume:

The rest is the same as for Blueberry row detection training and test pipelines.
