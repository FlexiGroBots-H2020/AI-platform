#!/bin/bash +x

MODEL_NAME="custom-model"
INPUT_PATH="@./input.json"
NAMESPACE="<NAMESPACE>"
SERVICE_HOSTNAME="https://${MODEL_NAME}-predictor-default-${NAMESPACE}.kubeflow.flexigrobots-h2020.eu"
COOKIE="<MTY3MzQ0 ... r6tXmgVPMeR>"

curl -v ${SERVICE_HOSTNAME}/v1/models/${MODEL_NAME}:predict \
	-d $INPUT_PATH \
	-H "Cookie: authservice_session=${COOKIE}"
