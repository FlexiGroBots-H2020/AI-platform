# Using inference services in the edge

Kubeflow's authentication system limits the use of KServe inference services
from outside the cluster by unauthenticated users/clients.
It is possible to overcome this limitation by using inference services over MQTT.
By doing this, an MQTT client can interact with the model by sending requests to
the server using a specific topic.
To do this, it is required to deploy an HTTP-MQTT *connector* (such as Apache Camel K),
and use an existing MQTT broker.
This guide compiles the required configuration steps.

## Mosquitto in Kubernetes (with Traefik)

This part of the guide describes how to deploy a
[Mosquitto](https://mosquitto.org/)
MQTT broker in Kubernetes using
[Traefik](https://traefik.io/)
as ingress controller (assuming it is already deployed and configured).
The applied configuration is partially based on the following articles:

- https://www.intuz.com/blog/how-to-deploy-mqtt-broker-in-a-kubernetes-cluster

- https://sko.ai/blog/how-to-run-ha-mosquitto/

### Extend Traefik configuration

First of all, it is required to modify the existing Traefik deployment
to create a new `entrypoint` for MQTT.
Edit the existing Traefik deployment as follows:

```diff
@@ -50,6 +50,7 @@
     spec:
       containers:
       - args:
         . . .
         - --entrypoints.web.address=:8000/tcp
         - --entrypoints.websecure.address=:8443/tcp
+        - --entrypoints.mqtt.address=:5883/tcp
         - --api.dashboard=true
         - --ping=true
         - --metrics.prometheus=true
@@ -83,6 +84,9 @@
         name: traefik
         ports:
         . . .
         - containerPort: 8443
           name: websecure
           protocol: TCP
+        - containerPort: 5883
+          name: mqtt
+          protocol: TCP
         readinessProbe:
           failureThreshold: 1
           httpGet:
```

In this project, port `5883` has been used, but this can be modified depending on the existing configuration.

### Mosquitto pre-configuration steps

Install `mosquitto` on your own host. This is just to complete the
configuration steps. Mosquitto broker will not be deployed this way.

```bash
$ sudo apt install mosquitto
```

Now, use `mosquitto_passwd` tool to generate desired MQTT-client credentials.
More information about this tool
[here](https://manpages.ubuntu.com/manpages/bionic/man1/mosquitto_passwd.1.html).

```bash
$ touch credentials.txt
$ mosquitto_passwd credentials.txt <USER>
```

This will fill `credentials.txt` file with the provided user name and the token
corresponding to the (requested) password. Example:

```bash
$ cat credentials.txt
admin:$6$84So0kRNW3ir+qd0$HFDYvYZ9CMM25ocu41YkTX2aXyGZBlkxFAblu2GRXh+/EMq88rEGX2nJYU+oJK2fzX8VzQK8wStEgMhHwjfHAg==
```
It is possible to run `mosquitto_passwd` tool over and over to generate more
client credentials.

### Kubernetes deployment

#### Configuring Mosquitto MQTT Broker

The desired configuration in Mosquitto is applied using two ConfigMaps
you can find in [./resources/mosquitto](./resources/mosquitto).

- [cm-mosquitto-config.yaml](./resources/mosquitto/cm-mosquitto-config.yaml):
  general configuration, following
  [mosquito.conf](https://github.com/eclipse/mosquitto/blob/master/mosquitto.conf).
  See https://mosquitto.org/man/mosquitto-conf-5.html for more information.

- [cm-mosquitto-password.yaml](./resources/mosquitto/cm-mosquitto-password.yaml),
  which has to be filled with the content of the `credentials.txt` file previously
  generated (note the indentation levels when adding content). Example:

  ```yaml
  apiVersion: v1
  kind: ConfigMap
  metadata:
    name: mosquitto-password
    namespace: mosquitto
  data:
    password.txt: |
      admin:$6$84So0kRNW3ir+qd0$HFDYvYZ9CMM25ocu41YkTX2aXyGZBlkxFAblu2GRXh+/EMq88rEGX2nJYU+oJK2fzX8VzQK8wStEgMhHwjfHAg==
  ```

#### Deploy Mosquitto MQTT Broker

In the appropriate K8s environment, create a dedicated namespace:

```bash
$ kubectl create ns mosquitto
```

Then apply K8s manifests:

```bash
$ kubectl apply -f ./resources/mosquitto/ -n mosquitto
```

After that, the broker should be ready and can be reachable by using its endpoint
(which will depend on both external DNS and Traefik configuration).

You can test whether everything works fine by using a desktop MQTT client such as
[MQTT Explorer](http://mqtt-explorer.com/).
