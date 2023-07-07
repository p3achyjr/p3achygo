rm -rf $1/chunks/
rm -rf $1/goldens/
find $1/models -type d -name 'model_*' ! -name 'model_0' -exec rm -rf {} +
rm -rf $1/sgf/
mkdir $1/chunks
mkdir $1/goldens/
mkdir $1/sgf
