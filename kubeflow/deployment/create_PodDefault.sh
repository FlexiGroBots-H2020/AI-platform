#!/bin/bash

if [ "$1" == "-h" ]; then
  echo "Usage: `basename $0` user-namespace"
  exit 0
elif [ $# -eq 0 ]; then
  echo "Error: no namespace provided"
  echo "Usage: `basename $0` user-namespace"
  exit 1
fi

sed 's/<YOUR_USER_PROFILE_NAMESPACE>/'$1'/g' 040-pod_default_multiuser.yaml \
	| kubectl apply -f -
