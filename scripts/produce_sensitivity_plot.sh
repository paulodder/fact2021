#!/bin/bash
CURRENT_DIR=$pwd
THIS_DIR="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $THIS_DIR
cd ..
source .env
cd $CURRENT_DIR
DATASET=$1
python $PROJECT_DIR/scripts/sensitivity_analysis.py --dataset $DATASET
python $PROJECT_DIR/scripts/visualize_sensitivity.py --dataset $DATASET
