# Example: how to create a custom Inference Service with KServe

More information can be found [here](https://kserve.github.io/website/0.7/modelserving/v1beta1/custom/custom_model/#deploy-locally-and-test).

- Build the docker image:

    ```bash
    $ docker build -t <REGISTRY>/custom-model:<VERSION> .
    ```

- Push the image to the appropriate registry:

    ```bash
    $ docker push <REGISTRY>/custom-model:<VERSION>
    ```

- Update the manifest of the example ```InferenceService``` [isvc.yaml](./isvc.yaml):

  ```yaml
  apiVersion: serving.kserve.io/v1beta1
  kind: InferenceService
  metadata:
    name: custom-model
  spec:
    predictor:
      containers:
      - name: kserve-container
        image: <REGISTRY>/custom-model:<VERSION>
        imagePullPolicy: Always
  ```

- Apply the manifest to deploy the ```InferenceService```:

    ```bash
    $ kubectl apply -f isvc.yaml [-n <YOUR_NAMESPACE>]
    ```

  Note: you do not need to specify the namespace when issuing ```kubectl``` commands
  in a terminal within a Kubeflow's Jupyter Notebook instance (since you are already in
  your own namespace).


- In the current instance of Kubeflow, the default ```InferenceService``` URLs
  are not accessible from outside de K8s cluster. Therefore, it is required to retrieve
  the valid endpoint for the ```InferenceService``` you have just deployed:

    ```bash
    $ kubectl get isvc <custom-model> -o jsonpath={.status.components.predictor.url} [-n <YOUR_NAMESPACE>]
    ```

- Use ```curl.sh``` script to test the deployed ```InferenceService```.

  Note: to test out an **external** prediction (sent from outside the cluster),
  the ```authservice_session``` cookie for the Kubeflow dashboard site
  from the browser must be added as a header in the request.

## Publishing inference results in MQTT

In the FlexiGroBots architecture, the deployment of inference services
is based on reaching them over MQTT.
More information can be found [here](../../../edge/README.md).
Before the inference services are able to publish inference results
to MQTT topics, some additional steps must be completed.

First, a K8s secret mus be created to store MQTT broker details
(ULR, port & credentials).
This secret must be created **in the same namespace
where the inference services will be deployed**.

```sh
$ kubectl create secret generic mqtt -n common-apps \
    --from-literal=BROKER_ADDRESS='<BROKER_ADDRESS>' \
    --from-literal=BROKER_PORT='<BROKER_PORT>' \
    --from-literal=BROKER_USER='<BROKER_USER>' \
    --from-literal=BROKER_PASSWORD='<BROKER_PASSWORD>'
```


It is possible to check whether the secret was created properly, e.g.:

```sh
$ kubectl get secrets -n common-apps mqtt -o jsonpath='{.data.address}' | base64 --decode
```

To create an new inference service including an MQTT client,
modify the following files as follows (examples):

- [requirements.txt](./requirements.txt):

  ```diff
  @@ -1,3 +1,4 @@
  kserve==0.7.0
  protobuf==3.20.*
  +paho-mqtt==1.6.1
  ```

- [inference_service_kserve.py](./inference_service_kserve.py):

  ```diff
  @@ -1,6 +1,9 @@
  import kserve
  from typing import Dict
  import logging
  +import os
  +import paho.mqtt.publish as publish
  +import json

  class Model(kserve.KFModel):
      def __init__(self, name: str):
  @@ -14,7 +17,18 @@ class Model(kserve.KFModel):
      def predict(self, request: Dict) -> Dict:
          logging.info("Payload: %s", request)

  -        return {"prediction": [1, 2, 3]}
  +        output = {"prediction": [1, 2, 3]}
  +
  +        publish.single("common-apps/{}/output".format(self.name),
  +                json.dumps(output),
  +                hostname=os.getenv('BROKER_ADDRESS'),
  +                port=int(os.getenv('BROKER_PORT')),
  +                client_id=self.name,
  +                auth = {"username": os.getenv('BROKER_USER'),
  +                        "password": os.getenv('BROKER_PASSWORD')}
  +            )
  +
  +        return {}
  ```

  Note that MQTT broker configuration is taken from environment
  variables that are read from the created K8s secret, which is loaded
  into the inference service resource (see next step).

- [isvc.yaml](./isvc.yaml):

  ```diff
  @@ -6,4 +6,16 @@ spec:
    predictor:
      containers:
        - name: kserve-container
  +        envFrom:
  +        - secretRef:
  +            name: mqtt
  ```

  Optional: it is also possible to define the resources to be used by the inference
  service (including the GPU) as follows:

  ```diff
  @@ -6,4 +6,16 @@ spec:
    predictor:
      containers:
        - name: kserve-container
          envFrom:
          - secretRef:
              name: mqtt
  +        resources:
  +          limits:
  +            cpu: "8"
  +            memory: 8Gi
  +            nvidia.com/gpu: "1"
  +          requests:
  +            cpu: "1"
  +            memory: 2Gi
  ```