if [ $# -lt 2 ]
then
    echo ./data.sh "<xc_dir> <data_name=G-LF-WikiSeeAlsoTitles-300K or G-LF-AmazonTitles-1.6M>"
    exit 1
fi

if [ ! -d $1 ]
then
    echo ERROR:: Directory does not exist, $1
    exit 1
fi

data_name=$2
if [ $data_name != "G-LF-WikiSeeAlsoTitles-300K" ] && [ $data_name != "G-LF-AmazonTitles-1.6M" ]
then
    echo Dataname should be in "G-LF-WikiSeeAlsoTitles-300K" and "G-LF-AmazonTitles-1.6M".
    exit 1
fi

echo python convert_xc_to_pecos.py --xc_dir=$1 --data_name=$data_name --save_dir=Datasets/
