# FlexiGroBots Artificial Intelligence platform

FlexiGroBots Artificial Intelligence (AI) platform will allow building, sharing and deploying AI services in the context of an agricultural data space.

It is initially based on [Kubeflow](https://www.kubeflow.org/).

## Install

FlexiGroBots AI platform uses Kubeflow v1.4.1. Therefore, the installation process is mostly based on the instructions for this software, which are described in [Installing Kubeflow](https://www.kubeflow.org/docs/started/installing-kubeflow/).

As can be seen in this page, installation guidelines are provided for multiple managed platforms and distributions, e.g., Amazon Web Services, Microsoft Azure, Google Cloud, etc. Nevertheless, within FlexiGroBots project, the goal is to have control over the complete stack. Therefore, installation will follow [Kubeflow Manifests] (https://github.com/kubeflow/manifests).

In particular, [instructions for v1.4.1](https://github.com/kubeflow/manifests/tree/v1.4.1) have been applied.

The installation instructions in this guide are written for Ubuntu 20.04.3 LTS. It must be adapted for a different operating system.

### Requirements

- [kustomize](https://kustomize.io/) (version 3.2.0).

```bash
curl -L https://github.com/kubernetes-sigs/kustomize/releases/download/v3.2.0/kustomize_3.2.0_linux_amd64 > kustomize
chmod +x kustomize
cp ./kustomize /usr/bin/
```

- kubectl

```bash
sudo apt-get install kubectl
```

- helm. Detailed instructions can be found [here](https://helm.sh/docs/intro/install/). 

- A Kubernetes cluster with a default StorageClass. Although in the official instructions v1.19 is recommended, FlexiGroBots installation has been done with v1.21.

#### Creating a Kubernetes cluster

The following steps describe the process to create a single-node Kubernetes cluster. These are the characteristics of the node:

- 4 cores
- RAM memory: 32 GB.
- HDD: 4TB.
- Operating system: Ubuntu 20.04 LTS
- Docker (20.10.12) and UFW (v0.36) installed.
- Open ports: 2379, 2049, 8080, 80.

For this guide, [Rancher 2.6](https://rancher.com/docs/rancher/v2.6/en/) has been used to create the Kubernetes cluster. 

- In the home page, select `Create`.
- Select the option, `Custom - Use existing nodes and create a cluster using RKE`.
- Add a name for the new cluster.
- Select as Kubernetes version `v1.21.9-racher-1-1`.
- Let the rest of the options with default values and press `Next`.
- Choose the roles for the first node in the cluster. Note that this part will strongly depend in the specific configuration of the target cluster. In this case, a single node cluster will be used. Therefore, select as `Node Role` the three options: `etcd`, `Control Plane` and `Worker`.
- Copy the command provided by Rancher and execute it in the node where Kubernetes will be installed.
- After some time, the cluster will be ready and it will be showed in Rancher user interface.

An additional configuration must be added in the cluster to avoid issues during Kubeflow installation.

- From `Cluster Management`, select `Edit Config` for the new cluster.
- Select the option `Edit as YAML`.
- Under `kube-controller:`, add the following lines:

```yaml
    kube-controller:
      extra_args:
        cluster-signing-cert-file: /etc/kubernetes/ssl/kube-ca.pem
        cluster-signing-key-file: /etc/kubernetes/ssl/kube-ca-key.pem
```

- Save the new configuration. It will take some time for the cluster to get updated.

#### Creating a Kubernetes cluster


### Installation process

## Additional configuration

### HTTP certificates

### Authentication