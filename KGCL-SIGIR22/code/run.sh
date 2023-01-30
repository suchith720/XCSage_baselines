if [ $# -lt 1 ]
then
    echo ./run "<dataset_name> <model>"
    exit
fi

dataset_name=$1
if [ $dataset_name != "G-LF-WikiSeeAlsoTitles-300K" ] && [ $dataset_name != "G-LF-AmazonTitles-1.6M" ]
then
    echo ERROR:: Dataset name should be - G-LF-AmazonTitles-1.6M or G-LF-WikiSeeAlsoTitles-300K
    exit
fi

model_type=$2
if [ $model_type != 'kgcl' ] && [ $model_type != 'lgn' ] && [ $model_type != 'sgl' ]
then
    echo ERROR:: Model type should be kgcl, lgn or sgl.
    exit
fi

python main.py --dataset=$dataset_name --topks=[3000] --model=$model_type --bpr_batch=1000 \
    --testbatch=1000 --epochs=10
