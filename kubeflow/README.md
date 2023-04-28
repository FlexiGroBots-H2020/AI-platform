# Kubeflow

Kubeflow is one of the subsystems that are part of FlexiGroBots AI platform.

## Install

FlexiGroBots AI platform uses Kubeflow v1.5.0. Therefore, the installation process is mostly based on the instructions for this software, which are described in [Installing Kubeflow](https://www.kubeflow.org/docs/started/installing-kubeflow/).

As can be seen in this page, installation guidelines are provided for multiple managed platforms and distributions, e.g., Amazon Web Services, Microsoft Azure, Google Cloud, etc. Nevertheless, within FlexiGroBots project, the goal is to have control over the complete stack. Therefore, installation will follow [Kubeflow Manifests] (https://github.com/kubeflow/manifests).

In particular, [instructions for v1.5.0](https://github.com/kubeflow/manifests/tree/v1.5.0) have been applied.

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

The following steps describe the process to create a two-node Kubernetes cluster. These are the characteristics of the nodes:

- Master:
  - CPU: 12 cores
  - RAM: 32GB
  - Disk: 4TB
  - Operating system: Ubuntu 20.04.2 LTS
  - Docker (20.10.12) and UFW (v0.36) installed
  - Open ports: [control plane and etcd ports](https://rancher.com/docs/rke/latest/en/os/#ports)

- Worker:
  - CPU: 64 cores
  - RAM: 124GB
  - Disk: 890GB
  - GPU: Tesla T4
  - Operating system: Ubuntu 20.04 LTS
  - Docker (20.10.17) and UFW (v0.36.1) installed
  - Open ports: [worker ports](https://rancher.com/docs/rke/latest/en/os/#ports)

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
- Choose the roles for the nodes in the cluster.
  Note that this part will strongly depend in the specific configuration of the target cluster.
  In this case, a two-node cluster will be used:
  - Select as `Node Role` the options: `etcd`, `Control Plane` and `Worker`.
  - Copy the command provided by Rancher and execute it in the master node where Kubernetes will be installed.
  - Then select as `Node Role` the options: `Worker`.
  - Copy the command provided by Rancher and execute it in the worker node where Kubernetes will be installed.
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
export KUBECONFIG=<PATH>
```

#### Installing a StorageClass

For the purpose of this guide, [NFS subdir external provisioner is being used](https://github.com/kubernetes-sigs/nfs-subdir-external-provisioner). A NFS server must be available.

The first step is to have our NFS server ready. How can we do that in a easy way? In our case our NFS server will be the master node. So to set up our NFS server in master node we need to execute in this master node (please refer to https://www.tecmint.com/install-nfs-server-on-ubuntu/ for more information):

```
// Install NFS Kernel Server
sudo apt install nfs-kernel-server

// Create an NFS Export Directory
sudo mkdir -p /nfsroot/kubeflow_nfs
sudo chown -R nobody:nogroup /nfsroot/kubeflow_nfs/
sudo chmod 777 /nfsroot/kubeflow_nfs/

// Grant NFS Share Access to Client Systems
sudo vim /etc/exports

// Add clients (IP of server in our kubernetes cluster) at the end of the file, for example: 
// /nfsroot/kubeflow_nfs  12.34.56.78/32(rw,no_root_squash,no_subtree_check)

// Export the NFS Share Directory
sudo exportfs -a
sudo systemctl restart nfs-kernel-server
sudo ufw allow from 12.34.56.78/32 to any port nfs
```

The installation of the provisioner is done with the following command:

```bash
helm install nfs-subdir-external-provisioner nfs-subdir-external-provisioner/nfs-subdir-external-provisioner \
    --set nfs.server=<NFS_SERVER_IP_ADDRESS> \
    --set nfs.path=<NFS_SERVER_PATH> \
    -n nfs-provisioner \
    --create-namespace
```

Mark the new `StorageClass` as default:

```bash
kubectl patch storageclass nfs-client -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'
```

If the previous command fails then execute:
```
kubectl edit storageclass nfs-client

// Add this line in "metadata" section
storageclass.kubernetes.io/is-default-class = "true"
```

If you want to test if NFS subdir external provisioner is working please execute:
```
kubectl create -f deploy/test-claim.yaml -f deploy/test-pod.yaml

// You can check in path /nfsroot/kubeflow_nfs/PVC-random-hash (in master node) that a file called SUCCESS was created. This means NFS subdir external provisioner is working correctly

// To delete this test
kubectl delete -f deploy/test-pod.yaml -f deploy/test-claim.yaml
```

Note:
if a node is added to the Kubernetes cluster after the NFS StorageClass has been installed,
you may need to modify configuration in ```/etc/exports``` to ensure access to NFS from the new node.
Example of adding new node with public IP ```5.79.113.30```:

```diff
--- /etc/exports     2022-08-01 07:10:20.114506376 +0000
+++ /etc/exports     2022-08-01 07:10:00.053658443 +0000
@@ -10,4 +10,5 @@
 #
 <NFS_SERVER_PATH> 213.227.145.0/16(rw,no_root_squash,no_subtree_check)
 <NFS_SERVER_PATH> 62.212.86.0/16(rw,no_root_squash,no_subtree_check)
+<NFS_SERVER_PATH> 5.79.113.0/16(rw,no_root_squash,no_subtree_check)
```

#### Installing MetalLB

Depending on the configuration applied during cluster creation (i.e.: enabling or not Nginx Ingress in Rancher), you may need to install MetalLB. MetalLB provides a load-balancer implementation which allows to create Kubernetes services of type `LoadBalancer` (more details [here](https://metallb.org/concepts/)).


The installation can be done using Helm by issuing the following commands (more details [here](https://metallb.org/installation/#installation-with-helm)):

```bash
helm repo add metallb https://metallb.github.io/metallb
helm install metallb metallb/metallb -n metallb-system --create-namespace
```

Then you need to provide the IP(s) to assign to `LoadBalancer` services (more details [here](https://metallb.org/configuration/)).
For example, the following configuration gives MetalLB control over AI-platform-cluster's public IP and configures Layer 2 mode:

```yaml
apiVersion: metallb.io/v1beta1
kind: IPAddressPool
metadata:
  name: ai-platform-pool
  namespace: metallb-system
spec:
  addresses:
  - XXX.XXX.XXX.XXX/32
---
apiVersion: metallb.io/v1beta1
kind: L2Advertisement
metadata:
  name: ai-platform
  namespace: metallb-system
```

Save the previous manifests as a `.yaml` file and apply them:

```bash
kubectl apply -f <manifests>.yaml
```

### Kubeflow installation

- Clone Kubeflow manifests, change to the appropriate branch and build Kubeflow configuration:

  ```batch
  git clone https://github.com/kubeflow/manifests.git
  git checkout tags/v1.5.0
  ```

- Modify configuration in `common/oidc-authservice/base/params.env` for OIDC provider. Example:

  ```diff
  --- a/common/oidc-authservice/base/params.env
  +++ b/common/oidc-authservice/base/params.env
  @@ -1,4 +1,4 @@
  -OIDC_PROVIDER=http://dex.auth.svc.cluster.local:5556/dex
  +OIDC_PROVIDER=https://kubeflow.flexigrobots-h2020.eu/dex
  OIDC_AUTH_URL=/dex/auth
  OIDC_SCOPES=profile email groups
  REDIRECT_URL=/login/oidc
  ```

- Modify configuration in `apps/centraldashboard/upstream/base/params.env` so that namespaces
  are created automatically when a new user uses the platform:

  ```diff
  --- a/apps/centraldashboard/upstream/base/params.env
  +++ b/apps/centraldashboard/upstream/base/params.env
  @@ -1,4 +1,4 @@
  CD_CLUSTER_DOMAIN=cluster.local
  CD_USERID_HEADER=kubeflow-userid
  CD_USERID_PREFIX=
  -CD_REGISTRATION_FLOW=false
  +CD_REGISTRATION_FLOW=true
  ```

- Build the Kubeflow configuration:

  ```bash
  kustomize build example > kubeflow.yaml
  ```

- Modify `istio-ingressgateway service` manifest in `kubeflow.yaml` file. Service `type` must be changed from `NodePort` to `LoadBalancer` to enable ingress traffic using cluster's external IP or AI-platform domain:

  ```diff
  @@ -122360,7 +122360,7 @@
    selector:
      app: istio-ingressgateway
      istio: ingressgateway
  -  type: NodePort
  +  type: LoadBalancer
  ---
  apiVersion: v1
  kind: Service
  ```

  Later on, when manifests are installed, the `EXTERNAL-IP` value of the service will be configured according to the configuration provided in `IPAddressPool` resource:

  ```bash
  $ kubectl get svc istio-ingressgateway -n istio-system
  NAME                   TYPE           CLUSTER-IP      EXTERNAL-IP       PORT(S)                                                                      AGE
  istio-ingressgateway   LoadBalancer   10.43.227.100   213.227.145.163   15021:32662/TCP,80:32297/TCP,443:31681/TCP,31400:31274/TCP,15443:32599/TCP   10m
  ```

- Modify `dex configmap` manifest in `kubeflow.yaml` to enable authentication based on GitHub. For this purpose, it is necessary to create an organization, and within it a team. Then you can choose the user account that has access to the platform. 
The values ClientID and ClientSecret can be obtained from organization settings in "Developer settings", and from this, we click on OAuth Group. We select Kubeflow dex, and in the new window, we can see the ClientID and ClientSecretID.



```diff
--- auth.yaml  2023-04-10 14:37:31.205665100 +0200
+++ auth.yaml  2023-04-10 14:30:41.536212400 +0200
@@ -5,14 +5,16 @@
 namespace: auth

 apiVersion: v1

 data:
   config.yaml: |
-    issuer: http://dex.auth.svc.cluster.local:5556/dex
+    issuer: https://kubeflow.project-example.com/dex
     storage:
       type: kubernetes
       config:
         inCluster: true
     web:
       http: 0.0.0.0:5556
     logger:
       level: "debug"
       format: text
+    connectors:
+      - type: github 
+        # Required field for connector id.
+        id: github
+        # Required field for connector name.
+        name: GitHub
+        config:
+          # Credentials can be string literals or pulled from the environment.
+          clientID: XXXXXXXXXXX
+          clientSecret: XXXXXXXXXXXXXXXXXXXX
+          redirectURI: https://kubeflow.project-example.com/dex/callback
+          orgs:
+            - name: repository/project-name
+              teams:
+                - team-name
+          loadAllGroups: true
+          useLoginAsID: true
     oauth2:
       skipApprovalScreen: true
-    enablePasswordDB: true
+    # enablePasswordDB is used to register with an e-mail. If you want to use a mail change to true
+    enablePasswordDB: false
-    staticPasswords:
-      - email: user@example.com
-        hash: ********
-        # https://github.com/dexidp/dex/pull/1601/commits
-        # FIXME: Use hashFromEnv instead
-        username: user
-        userID: "xxxxx"
     staticClients:
       - idEnv: OIDC_CLIENT_ID
         redirectURIs: ["/login/oidc"]
         name: 'Dex Login Application'
         secretEnv: OIDC_CLIENT_SECRET
```

  Note: for debugging purposes, `dex` manifests can be also built by issuing:

  ```bash
  kustomize build common/dex/overlays/istio  > dex.yaml
  ```

- Apply the manifest `.yaml` file:

  ```bash
  kubectl apply -f kubeflow.yaml
  ```

  As stated [here](https://github.com/kubeflow/manifests#prerequisites), the previous command
  may fail on the first try. This is inherent in how Kubernetes and kubectl work (e.g., CRs
  must be created after CRDs becomes ready). The solution is to simply re-run the command
  until it succeeds.

  It will take some time to deploy all the pods. You can check that everything is running:

  ```bash
  kubectl get pods -n cert-manager
  kubectl get pods -n istio-system
  kubectl get pods -n auth
  kubectl get pods -n knative-eventing
  kubectl get pods -n knative-serving
  kubectl get pods -n kubeflow
  ```

  Notes:
  - If the `OIDC_PROVIDER` configuration endpoint is intended to use TLS (https) the `authservice-0 `
    pod will not be ready until TLS configuration is completed (see this [section](#https-certificate)).
  - In the particular case of the Kubeflow deployment for the FlexiGroBots project,
    it was found that some pods did not became `Running` and logs outputted
    `"unable to run the manager","error":"too many open files"`.
    The issue was resolved by issuing the following commands (more information can be found
    [here](https://github.com/kubeflow/manifests/issues/2087)):

    ```bash
    sudo sysctl fs.inotify.max_user_instances=1280
    sudo sysctl fs.inotify.max_user_watches=655360
    ```

- In order to check the access to the Kubeflow's dashboard, port `80` of Istio `Ingress-Gateway service`
  can be forwarded to the host:

  ```bash
  kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80
  ```

  Then, the dashboard will be available in [http://localhost:8080](http://localhost:8080).
  Default credentials are `user@example.com` and `12341234`.

  Notes:
  - This will not work if the `OIDC_PROVIDER` is intended to use TLS until TLS configuration
    is completed (see this [section](#https-certificate)).
  - This will not work either if default user creation is disabled in `dex` `configmap` manifest.

### Additional configuration

#### HTTPS certificate

A DNS domain should be available and configured for the IP address of the Kubernetes node where the `Istio` ingress will be deployed.

**Option A: generate certificates using Let's Encrypt** (not currently in use)

An HTTPS certificate can be obtained from a provider or from [Let's Encrypt](https://letsencrypt.org/). The following steps can be used to automate the retrieval of a certificate:

- Create two `Issuer` objects for staging and production environments.

  ```bash
  kubectl apply -f deployment/010-issuer.yml
  ```

- Create two `Certificate` objects for staging and production environments.

  ```bash
  kubectl apply -f deployment/020-certificates.yml
  ```

- Apply the new configuration for Istio Ingress-Gateway:

  ```bash
  kubectl apply -f deployment/030-gw_https.yaml
  ```

__Important:__

The process of issuing certificates will not work by default and it will remain in
"Issuing certificate as Secret does not exist" status because "challenges"
will not be able to complete.
A “challenge” is the method Let’s Encrypt servers use to validate that you control
the domain names you are adding to certificates.
(more details can be found [here](https://letsencrypt.org/docs/challenge-types/#http-01-challenge)).

In summary, Let’s Encrypt gives a token to your ACME client, and your ACME client puts
a file on your web server at `http://<DOMAIN>/.well-known/acme-challenge/<TOKEN>`.
The problem is that an Istio's `Envoyfilter` resource called `authn-filter` is blocking
communications towards that `url` and hence the challenge is never completed.

A workaround to overcome this issue is to disable the `Envoyfilter` mentioned above,
let the challenges complete and finally re-enable the `Envoyfilter` again:

```bash
kubectl get envoyfilters.networking.istio.io authn-filter -n istio-system -o yaml > envoyfilter.yaml
kubectl delete envoyfilters.networking.istio.io authn-filter -n istio-system
```

After a few minutes, challenges will be completed and the certificate status will read
"Certificate is up to date and has not expired".
Then, re-deploy the `Envoyfilter`:

```bash
kubectl apply -f envoyfilter.yaml
```

**Option B: use manually-generated certificates**

- Create a secret with TLS certificate and key for the ingress gateway:

  ```bash
  kubectl create -n istio-system secret tls kubeflow-domain-man-cert \
    --key=<PATH_TO_CERTS>.key \
    --cert=<PATH_TO_CERTS>.crt
  ```

- Apply the new configuration for Istio Ingress-Gateway:

  ```bash
  kubectl apply -f deployment/031-gw_https_manual_certificates.yaml
  ```

#### Enabling access to Kubeflow Pipelines

Once a user has logged in, a `PodDefault` resource must be created in user's
namespace to enable the pipeline upgrade process from Jupyter Notebooks.
Apply the `deployment/040-pod_default_multiuser.yaml` manifest replacing in each case
the name of the corresponding namespace:

```yaml
apiVersion: kubeflow.org/v1alpha1
kind: PodDefault
metadata:
  name: access-ml-pipeline
  namespace: "<YOUR_USER_PROFILE_NAMESPACE>"
spec:
...
```

```bash
kubectl apply -f deployment/040-pod_default_multiuser.yaml
```

This task can be automated by running the `deployment/create_PodDefault.sh` script:

```bash
./create_PodDefault.sh <USER_PROFILE_NAMESPACE>
```

## Using GPU

In order to use an Nvidia GPU with Kubernetes and Kubeflow, [NVIDIA Cloud Native Documentation](https://docs.nvidia.com/datacenter/cloud-native/contents.html) must be installed.

The following instructions come directly from this page.

Note: NVIDIA driver must be installed for the Linux distribution in use.
Check it out by issuing ```nvidia-msi```.
If the output is similar to the following,
install  the driver before getting started (the recommended way to install
it is to use the package manager of the Linux distribution):

```bash
Command 'nvidia-smi' not found, but can be installed with:

sudo apt install nvidia-340               # version 340.108-0ubuntu5.20.04.2, or
sudo apt install nvidia-utils-390         # version 390.151-0ubuntu0.20.04.1
sudo apt install nvidia-utils-450-server  # version 450.191.01-0ubuntu0.20.04.1
sudo apt install nvidia-utils-470         # version 470.129.06-0ubuntu0.20.04.1
sudo apt install nvidia-utils-470-server  # version 470.129.06-0ubuntu0.20.04.1
```

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
```

At this point, a working setup can be tested by running a base CUDA container:

```bash
sudo docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
```

This should result in a console output shown below:

```bash
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

Since NVIDIA drivers have been installed in the previous step, NVIDIA GPU operator will be installed setting ```driver.enabled``` to ```false```:

```bash
helm install --wait --generate-name -n gpu-operator --create-namespace nvidia/gpu-operator --set driver.enabled=false
```

#### Time-slicing (optional)

NVIDIA GPUs are now schedulable resources in Kubernetes.
However, the previous configuration only allows for devices –including GPUs (as nvidia.com/gpu)–
to be advertised as integer resources in Kubernetes and thus does not allow for oversubscription
–this is, sharing the same GPU unit among different K8s resources.
In this [article](https://developer.nvidia.com/blog/improving-gpu-utilization-in-kubernetes/)
further information about oversubscribing GPUs in Kubernetes using time-slicing is explained.
In summary, the following steps are required:

Modify ```deployment/gpu-time-slicing-config.yaml``` to match the number of nvidia.com/gpu
resources you want to advertise to Kubernetes:

```diff
--- a/kubeflow/deployment/gpu-time-slicing-config.yaml
+++ b/kubeflow/deployment/gpu-time-slicing-config.yaml
@@ -16,4 +16,4 @@ sharing:
   timeSlicing:
     resources:
     - name: nvidia.com/gpu
-      replicas: 2
+      replicas: <4>
```

Then create a ```ConfigMap``` containing that configuration file:

```bash
kubectl create configmap time-slicing --from-file deployment/gpu-time-slicing-config.yaml -n gpu-operator
```

Finally, upgrade Operator's deployment.
You will need to include your Operator's release name in the command below:

```bash
helm upgrade --wait \
    -n gpu-operator \
    gpu-operator-<1660041098> nvidia/gpu-operator \
    --set driver.enabled=false \
    --set devicePlugin.config.name=time-slicing \
    --set devicePlugin.config.default=gpu-time-slicing-config.yaml
```

If the changes are applied successfully, inspecting the status of the node
containing the physical GPU will show that the device plugin advertises
the configured number of GPUs as allocatable:

```bash
kubectl describe node <GPU-node-name>

...
Capacity:
  cpu:                64
  ephemeral-storage:  102626232Ki
  hugepages-1Gi:      0
  hugepages-2Mi:      0
  memory:             131363284Ki
  nvidia.com/gpu:     2
  pods:               110
Allocatable:
  cpu:                64
  ephemeral-storage:  94580335255
  hugepages-1Gi:      0
  hugepages-2Mi:      0
  memory:             131260884Ki
  nvidia.com/gpu:     2
  pods:               110
```