#!/usr/bin/env bash

LOG_ALIAS=$1
LOG_DIR="logs"
mkdir -p ${LOG_DIR}

LOG_FILE="${LOG_DIR}/${LOG_ALIAS}_`date +'%Y-%m-%d_%H:%M.%S'`.log"
#echo $LOG_FILE

./train.py --arch="mobilenet_1" \
    --start-epoch=1 \
    --loss=vdc \
    --snapshot="snapshot/phase1_pdc" \
    --param-fp-train='../train.configs/param_all_norm.pkl' \
    --param-fp-val='../train.configs/param_all_norm_val.pkl' \
    --warmup=5 \
    --opt-style=resample \
    --batch-size=256 \
    --base-lr=0.01 \
    --epochs=50 \
    --milestones=30,40 \
    --print-freq=50 \
    --devices-id=0,1,2,3 \
    --workers=8 \
    --filelists-train="../train.configs/train_aug_120x120.list.train" \
    --filelists-val="../train.configs/train_aug_120x120.list.val" \
    --root="/path/to//train_aug_120x120" \
    --log-file="${LOG_FILE}"
