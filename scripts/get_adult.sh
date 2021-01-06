CURRENT_DIR=$pwd
THIS_DIR="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $THIS_DIR
cd ..
source .env
cd  $DATA_DIR
cd $CURRENT_DIR
mkdir $DATA_DIR/adult
wget http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data -O $DATA_DIR/adult/train.csv
wget http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test -O $DATA_DIR/adult/test.csv
