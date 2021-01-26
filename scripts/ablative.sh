#!/bin/bash
CURRENT_DIR=$pwd
THIS_DIR="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $THIS_DIR
cd ..
source .env
cd $CURRENT_DIR
echo $RESULTS_DIR
DATASET=$1
echo $DATASET
SEEDS=" 42 420 4200 0 1 "
LOSS_COMPS=" none entropy kl,orth entropy,kl entropy,kl,orth "
for l in $LOSS_COMPS
do
    for s in $SEEDS
    do
        echo "seed" $s
        echo "loss comps" $l
        python $PROJECT_DIR/scripts/train.py --dataset $DATASET --experiment ablative --loss_components $l --seed $s
    done
done
