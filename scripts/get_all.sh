CURRENT_DIR=$pwd
THIS_DIR="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
bash $THIS_DIR/get_adult.sh
bash $THIS_DIR/get_german.sh
bash $THIS_DIR/get_cifar10.sh
bash $THIS_DIR/get_cifar100.sh
bash $THIS_DIR/get_yaleb.sh
