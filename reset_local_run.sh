rm -rf $1/chunks/
rm -rf $1/goldens/
find $1/models -type d -name 'model_*' ! -name 'model_0000' -exec rm -rf {} +
rm -rf $1/sgf/
rm -f $1/batch_num.txt
rm -f $1/elo_history.txt
mkdir $1/chunks
mkdir $1/goldens/
mkdir $1/sgf
