CURRENT_DIR=$pwd
THIS_DIR="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $THIS_DIR
cd ..
source .env
cd $DATA_DIR
cd $CURRENT_DIR
mkdir $DATA_DIR/cifar10
wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -O $DATA_DIR/cifar10/cifar10.tar.gz
tar -xvf $DATA_DIR/cifar10/cifar10.tar.gz -C $DATA_DIR/cifar10
rm $DATA_DIR/cifar10/cifar10.tar.gz
