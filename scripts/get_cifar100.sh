CURRENT_DIR=$pwd
THIS_DIR="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $THIS_DIR
cd ..
source .env
cd $DATA_DIR
cd $CURRENT_DIR
mkdir $DATA_DIR/cifar100
wget http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz -O $DATA_DIR/cifar100/cifar100.tar.gz
tar -xvf $DATA_DIR/cifar100/cifar100.tar.gz -C $DATA_DIR/cifar100
rm $DATA_DIR/cifar100/cifar100.tar.gz
