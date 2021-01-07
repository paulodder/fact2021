CURRENT_DIR=$pwd
THIS_DIR="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $THIS_DIR
cd ..
source .env
cd $DATA_DIR
cd $CURRENT_DIR
mkdir $DATA_DIR/yaleb
wget http://vision.ucsd.edu/extyaleb/CroppedYaleBZip/CroppedYale.zip -O $DATA_DIR/yaleb/yaleb.zip
unzip $DATA_DIR/yaleb/yaleb.zip -d $DATA_DIR/yaleb
rm $DATA_DIR/yaleb/yaleb.zip
