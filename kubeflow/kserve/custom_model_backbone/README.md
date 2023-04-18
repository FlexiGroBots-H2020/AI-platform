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

- Update the manifest of the example ```InferenceService```:

  ```isvc.yaml```:
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