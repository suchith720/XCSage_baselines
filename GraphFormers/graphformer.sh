if [ $# -lt 1 ]
then
    echo ./graphformer.sh '<main_dir>'
    exit
fi

if [ ! -d $1 ]
then
    echo Directory does not exist, $1
    exit
fi

./data.sh $1 G-LF-WikiSeeAlsoTitles-300K
./run.sh train $1 G-LF-WikiSeeAlsoTitles-300K
./run.sh test $1 G-LF-WikiSeeAlsoTitles-300K

./data.sh $1 G-LF-AmazonTitles-1.6M similar_
./run.sh train $1 G-LF-AmazonTitles-1.6M similar_
./run.sh test $1 G-LF-AmazonTitles-1.6M similar_

./data.sh $1 G-LF-AmazonTitles-1.6M also_view_
./run.sh train $1 G-LF-AmazonTitles-1.6M also_view_
./run.sh test $1 G-LF-AmazonTitles-1.6M also_view_