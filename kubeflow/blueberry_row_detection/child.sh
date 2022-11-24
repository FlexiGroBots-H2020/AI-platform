#!/bin/bash
# module load cuda/11.2
# module load python/anaconda3

# export NCCL_LL_THRESHOLD=0
# SUBNET=10.30.0 # MILEVA specific

# MASTER_FILE=$SLURM_JOBID.master
# if [ $SLURM_PROCID -eq 0 ] ; then
# IP=$(ifconfig | grep $SUBNET | awk '{ print $2 }')
# echo "Master IP addr $IP"
# echo $SLURM_NPROCS
# MASTER="tcp://$IP:12345"
# echo $MASTER > $MASTER_FILE
# else
# while [ ! -f $MASTER_FILE ] ; do
# sleep 1
# done
# MASTER=$(cat $MASTER_FILE)
# fi
# env NCCL_LL_THRESHOLD=0

/home/stefanovicd/.conda/envs/torch_new_env/bin/python /home/stefanovicd/DeepSleep/BlueberryRowDetectionKubeflow/Train_BGFG_BCE_with_weightsUnet3.py
# /home/stefanovicd/.conda/envs/torch_new_env/bin/python /home/stefanovicd/DeepSleep/agrovision/BorovniceUnetBS/Train_BGFG_BCE_with_weights.py
# /home/stefanovicd/.conda/envs/torch_new_env/bin/python /home/stefanovicd/DeepSleep/agrovision/BorovniceUnetBS/Train_BGFG_BCE_with_weightsUnet3.py
# /home/stefanovicd/.conda/envs/torch_new_env/bin/python /home/stefanovicd/DeepSleep/agrovision/AgroVisionUnetBS/Clases_weights_calculation.py
#/storage/opt/anaconda3/bin/python
# $MASTER # Replace mnist.py with the path to your training scripta
# /home/stefanovicd/.conda/envs/tensorflow_novi/bin/python train.py
