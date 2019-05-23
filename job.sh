#!/bin/bash
machine=$1
gpu=$2
job=$3

ssh bolt$machine \
    source $HOME/export_p1.sh\; \
    cd $HOME/nuq/code/\;\
    CUDA_VISIBLE_DEVICES=$gpu sh jobs/machine"$machine"_gpu"$gpu"_job"$job".sh
