if [ $# -lt 1 ]
then
    echo ./elias.sh "<xc_dir>"
    exit 1
fi

tf_max_len="32"
tf_token_type="bert-base-uncased"

dataset_name="G-LF-WikiSeeAlsoTitles-300K"
echo Creating dataset $dataset_name
./data.sh $1 $dataset_name
./prepare.sh $dataset_name $tf_max_len $tf_token_type
echo "Running code"
./run_benchmark.sh $dataset_name

dataset_name="G-LF-AmazonTitles-1.6M"
echo Creating dataset $dataset_name
./data.sh $1 $dataset_name
./prepare.sh $dataset_name $tf_max_len $tf_token_type
echo "Running code"
./run_benchmark.sh $dataset_name
