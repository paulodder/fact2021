#!/bin/bash
CURRENT_DIR=$pwd
THIS_DIR="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $THIS_DIR
cd ..
source .env
cd $CURRENT_DIR
# adult
bash $PROJECT_DIR/scripts/ablative.sh adult
python $PROJECT_DIR/scripts/visualize_ablative.py -d adult
# german
bash $PROJECT_DIR/scripts/ablative.sh german
python $PROJECT_DIR/scripts/visualize_ablative.py -d german
# yaleb
bash $PROJECT_DIR/scripts/ablative.sh yaleb
python $PROJECT_DIR/scripts/visualize_ablative.py -d yaleb
# cifar10
bash $PROJECT_DIR/scripts/ablative.sh cifar10
python $PROJECT_DIR/scripts/visualize_ablative.py -d cifar10
# cifar100
bash $PROJECT_DIR/scripts/ablative.sh cifar100
python $PROJECT_DIR/scripts/visualize_ablative.py -d cifar100
