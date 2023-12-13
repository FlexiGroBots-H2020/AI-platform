# Delineation and zoning - Inference service

The pipeline consists of three components or blocks. Again, we must first download data from the data storage (minIO), then we have to choose a part of parcel where we want to calculate vegetation indices and generate maps that are further being used in ML pipeline for delineation and zoning. Lastly we need to provide the parcel zones back to our data storage in tif, shp or any other suitable format.

#### Step 1: Creating notebook
#### Step 2: Uploading yaml files for each component

YAML files for this pipeline can be found in FlexiGroBots [Github](https://github.com/FlexiGroBots-H2020/AI-platform/tree/biosens/kubeflow/Delineation%20Inference) repo in the corresponding folders under the biosens branch. Each file assigns appropriate docker image/package:
- Download component: rebuild_delineation_download_component 
- Test component: rebuild_delineation_test_component
- PostProcessing component: rebuild_delineation_postprocessing_component.

These docker images were pushed to the project GitHub repo and can be found in [packages](https://github.com/orgs/FlexiGroBots-H2020/packages) section.

An IPYNB file is also necessary and it is included in the same folders in GitHub repo. This file is essential for connecting individual components. It consists of at least two major cells. In the first one it’s always important to check volume name in the code and change it to match with corresponding volume:

The rest is the same as for Blueberry row detection training and test pipelines.
