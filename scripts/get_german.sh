CURRENT_DIR=$pwd
THIS_DIR="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $THIS_DIR
cd ..
source .env
cd  $DATA_DIR
cd $CURRENT_DIR
mkdir $DATA_DIR/german
wget https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data -O $DATA_DIR/german/data.csv
