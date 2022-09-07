# Kubeflow

Kubeflow is one of the subsystems that are part of FlexiGroBots AI platform.

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

- 64 cores
- RAM memory: 124GB.
- 2x960GB SSD, RAID 0.
- Operating system: Ubuntu 20.04 LTS
- Docker (20.10.12) and UFW (v0.36) installed.
- Open ports: 2379, 2049, 8080, 80.

The installation would be possible with a less powerful machine. For instance:

- 4 cores
- RAM memory: 32 GB.
- HDD: 4TB.

For this guide, [Rancher 2.6](https://rancher.com/docs/rancher/v2.6/en/) has been used to create the Kubernetes cluster. 

- In the home page, select `Create`.
- Select the option, `Custom - Use existing nodes and create a cluster using RKE`.
- Add a name for the new cluster.
- Select as Kubernetes version `v1.21.12`.
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
- Download the KubeConfig file
- Set KUBECONFIG environmental variable to the path of the KubeConfig file:
  
```bash
export KUBECONFIG=<path>
```

#### Installing a StorageClass

For the purpose of this guide, [NFS subdir external provisioner is being used](https://github.com/kubernetes-sigs/nfs-subdir-external-provisioner). A NFS server must be available.

The installation of the provisioner is done with the following command:

```bash
helm install nfs-subdir-external-provisioner nfs-subdir-external-provisioner/nfs-subdir-external-provisioner --set nfs.server=<NFS_SERVER_IP_ADDRESS> --set nfs.path=<NFS_SERVER_PATH>
```

Mark the new `StorageClass` as default:

```bash
kubectl patch storageclass nfs-client -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'
```

### Kubeflow installation

- Clone Kubeflow manifests, change to the appropriate branch and build Kubeflow configuration:

```batch
git clone https://github.com/kubeflow/manifests.git
git checkout tags/v1.5.0
```

Modify configuration file for OIDC provider. In `common/oidc-authservice/base/params.env`:

```yaml
OIDC_PROVIDER=https://<KUBEFLOW DOMAIN>/dex
```

Modify configuration file so that namespaces are created automatically when a new user uses the platform. In `apps/centraldashboard/upstream/base/params.env`:

```yaml
CD_REGISTRATION_FLOW=true
```

- Build the Kubeflow configuration:

```
kustomize build example > kubeflow.yaml
```

- Apply the yaml file:

```bash
kubectl apply -f kubeflow.yaml
```

It will take some time to deploy all the pods. You can check that everything is running:

```bash
kubectl get pods -n cert-manager
kubectl get pods -n istio-system
kubectl get pods -n auth
kubectl get pods -n knative-eventing
kubectl get pods -n knative-serving
kubectl get pods -n kubeflow
```

In order to access the dashboard, port 80 of Istio Ingress-Gateway must be forwarded to the host:

```bash
kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80
```

Then, the dashboard will be available in [`](http://localhost:8080)`. Default credentials are `user@example.com` and `12341234`.

### Additional configuration

#### Automatic profile creation

When a new user get access to Kubeflow, profile and namespace must be created.

For this, set `REGISTRATION_FLOW ` to `TRUE` by using:

```bash
kubectl edit deployment centraldashboard -n kubeflow
kubectl rollout restart deployments/centraldashboard -n kubeflow
```

#### HTTP certificate

A DNS domain should be available and configured for the IP address of the Kubernetes node where the `Istio` ingress will be deployed.

An HTTPS certificate can be obtained from a provider or from [Let's Encrypt](https://letsencrypt.org/). The following steps can be used to automate the retrieval of a certificate:

- Create two `Issuer` objects for staging and production environments.

```bash
kubectl apply -f 010-issuer.yml
```

- Create two `Certificate` objects for staging and production environments.

```bash
kubectl apply -f 020-certificates.yml
```

- Apply the new configuration for Istio Ingress-Gateway:

```bash
kubectl apply -f deployment/gw_https.yaml
```

#### Authentication

Authentication will rely on GitHub. In addition, a particular organisation's team has been allowed to access the kubeflow pod.  

- Build only Dex configuration:

```bash
kustomize build common/dex/overlays/istio  > dex.yaml
```

- Modify dex.yaml with the following content:

```yaml
data:
  config.yaml: |
    issuer: <KUBEFLOW_DOMAIN>/dex
    storage:
      type: kubernetes
      config:
        inCluster: true
    web:
      http: 0.0.0.0:5556
    logger:
      level: "debug"
      format: text
    connectors:
    - type: github
      # Required field for connector id.
      id: github
      # Required field for connector name.
      name: GitHub
      config:
        # Credentials can be string literals or pulled from the environment.
        clientID: <GIHUB_CLIENT_ID>
        clientSecret: <GIHUB_CLIENT_SECRET>
        redirectURI: <KUBEFLOW_DOMAIN>/dex/callback
        orgs:
        - name: Organization's name
          team:
          - Team's name
        loadAllGroups: true
        useLoginAsID: true

    oauth2:
      skipApprovalScreen: true
    # enablePasswordDB is used to log-in with an e-mail. If you want to use a mail change to true
    enablePasswordDB: false
    staticPasswords:
    - email: user@example.com
      hash: $2y$12$kAJmOQmkeaq5lNN8z3v9E.rS8cvd8Rm8MR3EbcWDEwPsFqq8mbpFS
      # https://github.com/dexidp/dex/pull/1601/commits
      # FIXME: Use hashFromEnv instead
      username: user
      userID: "15841185641784"
    staticClients:
    # https://github.com/dexidp/dex/pull/1664
    - idEnv: OIDC_CLIENT_ID
      redirectURIs: ["/login/oidc"]
      name: 'Dex Login Application'
      secretEnv: OIDC_CLIENT_SECRET
```

## Using GPU

In order to use an Nvidia GPU with Kubernetes and Kubeflow, [NVIDIA Cloud Native Documentation](https://docs.nvidia.com/datacenter/cloud-native/contents.html) must be installed.

The following instructions come directly from this page.

### Setting up NVIDIA Container Toolkit

In the node with the GPU the following commands must be applied.

Setup the package repository and the GPG key:

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

Install the `nvidia-docker2` package (and dependencies) after updating the package listing:

```bash
sudo apt-get update
sudo apt-get install -y nvidia-docker2
```

Restart the Docker daemon to complete the installation after setting the default runtime:

```bash
sudo systemctl restart docker
At this point, a working setup can be tested by running a base CUDA container:
```

```bash
sudo docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
```

```bash
This should result in a console output shown below:

+-----------------------------------------------------------------------------+
| NVIDIA-SMI 450.51.06    Driver Version: 450.51.06    CUDA Version: 11.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla T4            On   | 00000000:00:1E.0 Off |                    0 |
| N/A   34C    P8     9W /  70W |      0MiB / 15109MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

### Install NVIDIA GPU Operator

Add the NVIDIA Helm repository:

```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia \
   && helm repo update
```

Since NVIDIA drivers have been installed in the previous step, NVIDIA GPU operator will be installed indicating this option.

```bash
 helm install --wait --generate-name      -n gpu-operator --create-namespace      nvidia/gpu-operator      --set driver.enabled=false
```