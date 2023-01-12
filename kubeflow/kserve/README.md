# Example: how to create an Inference Service with KServe

- Apply the manifest of the example ```InferenceService``` named "flower-sample":

    ```bash
    $ kubectl apply -f isvc_example.yaml [-n <YOUR_NAMESPACE>]
    ```

  Note: you do not need to specify the namespace when issuing ```kubectl``` commands
  in a terminal within a Jupyter Notebook instance (since you are already in
  your own namespace).

  ```isvc_example.yaml```:
    ```yaml
    apiVersion: "serving.kserve.io/v1beta1"
    kind: "InferenceService"
    metadata:
    name: "flower-sample"
    spec:
    predictor:
        tensorflow:
        storageUri: "gs://kfserving-examples/models/tensorflow/flowers"
    ```

- In the current instance of Kubeflow, the default ```InferenceService``` URLs
  are not accessible from outside de K8s cluster. Therefore, it is required to retrieve
  the valid endpoint for the ```InferenceService``` you have just deployed:

    ```bash
    $ kubectl get isvc flower-sample -o jsonpath={.status.components.predictor.url} [-n <YOUR_NAMESPACE>]

    http://flower-sample-predictor-default-<name-space>.kubeflow.flexigrobots-h2020.eu
    ```

- Use this URL to make a test request. Note the request will use TLS (http**s**).
  As included [here](https://www.kubeflow.org/docs/distributions/ibm/deploy/authentication/),
  to test out an **external** prediction (sent from outside the cluster),
  the ```authservice_session``` cookie for the Kubeflow dashboard site
  from the browser must be added as a header in the request:

    ```bash
    $ curl -v https://flower-sample-predictor-default-<name-space>.kubeflow.flexigrobots-h2020.eu/v1/models/flower-sample:predict \
        -d @./input.json \
        -H "Cookie: authservice_session=MTY3MTEwNDQwN ... ilbl3p8efA"

    {
        "predictions": [
            {
                "scores": [0.999114931, 9.20987877e-05, 0.000136786475, 0.000337258185, 0.000300532876, 1.84813962e-05],
                "prediction": 0,
                "key": "   1"
            }
        ]
    }
    ```

## Custom inference services

Please refer to ```custom_model_backbone``` example and its documentation.