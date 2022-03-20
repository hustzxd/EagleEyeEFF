#!/usr/bin/env bash

# export PATH="/home/${USER}/anaconda3/bin:$PATH"
export CURRENT_DIR=`pwd`
export PYTHONPATH=$PYTHONPATH:${CURRENT_DIR}

protoc -I=${CURRENT_DIR}/proto --python_out=${CURRENT_DIR}/proto ${CURRENT_DIR}/proto/efficient_pytorch.proto
python proto/gene_hyperparam_template.py