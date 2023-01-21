if [ $# -lt 1 ]
then
    echo ./run "<dataset_name>"
    exit
fi

dataset_name=$1
if [ $dataset_name != "G-LF-WikiSeeAlsoTitles-300K" ] && [ $dataset_name != "G-LF-AmazonTitles-1.6M" ]
then
    echo ERROR:: Dataset name should be - G-LF-AmazonTitles-1.6M or G-LF-WikiSeeAlsoTitles-300K
    exit
fi

python main.py --dataset=$dataset_name --topks=[3000]
