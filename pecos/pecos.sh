if [ $# -lt 1 ]
then
    echo ./pecos.sh "<xc_dir>"
    exit 1
fi

if [ ! -e $1 ]
then
    echo Directory does not exist, $1
    exit 1
fi

data_name=G-LF-WikiSeeAlsoTitles-300K
./data.sh $1 $data_name
./run.sh $data_name

data_name=G-LF-AmazonTitles-1.6M
./data.sh $1 $data_name
./run.sh $data_name
