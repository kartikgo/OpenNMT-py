#!/bin/bash
GPU=$1
export CUDA_VISIBLE_DEVICES = $GPU
DATA=ted_data/de-en
REP=10000
SAVE=$2
mkdir -p $SAVE
LOGS=$SAVE/log
VALIDSTEPS=2000
TRAINSTEPS=100000
SNORM=0.0
if [ "$3" != "" ]; then
    VALIDSTEPS=$3
if [ "$4" != "" ]; then
    TRAINSTEPS=$4
if [ "$5" != "" ]; then
    SNORM=$5
python train.py -data $DATA -encoder_type brnn -report_every $REP -log_file $LOGS -save_model $SAVE/model  -gpu_ranks 0 -param_init 0.01 -batch_size 256 -valid_steps $VALIDSTEPS  -train_steps $TRAINSTEPS -optim adam -learning_rate 0.001 -learning_rate_decay 0.99 -snorm $SNORM
