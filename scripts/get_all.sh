CURRENT_DIR=$pwd
THIS_DIR="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
sh $THIS_DIR/get_adult.sh
sh $THIS_DIR/get_german.sh
sh $THIS_DIR/get_cifar10.sh
sh $THIS_DIR/get_cifar100.sh
sh $THIS_DIR/get_yaleb.sh
