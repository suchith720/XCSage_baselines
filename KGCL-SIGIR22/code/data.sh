if [ $# -lt 3 ]
then
    echo ./data.sh "<main_dir> <dataset_name> <graph_type> <save_dir:optional>"
    exit
fi

dataset_name=$2
if [ $dataset_name != "G-LF-WikiSeeAlsoTitles-300K" ] && [ $dataset_name != "G-LF-AmazonTitles-1.6M" ]
then
    echo ERROR:: Dataset name should be - G-LF-AmazonTitles-1.6M or G-LF-WikiSeeAlsoTitles-300K
    exit
fi

main_dir=$1/$dataset_name
if [ ! -d $main_dir ]
then
    echo Directory does not exist, $main_dir
    exit
fi

graph_type=$3

if [ $# -gt 3 ]
then
    save_dir=$4
else
    save_dir=../data/$dataset_name/
fi

mkdir -p $save_dir

python convert_xc_to_kgcl.py --xc_dir=$main_dir --save_dir=$save_dir \
    --graph_type=$graph_type
