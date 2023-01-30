if [ $# -lt 1 ]
then
    echo ./kgcl.sh "<main_dir> <model_type>"
    exit
fi

main_dir=$1
if [ ! -d $main_dir ]
then
    echo Directory does not exist, $main_dir
    exit
fi

model_type=$2

#./data.sh $main_dir G-LF-WikiSeeAlsoTitles-300K graph,category
./run.sh G-LF-WikiSeeAlsoTitles-300K $model_type

#./data.sh $main_dir G-LF-AmazonTitles-1.6M similar_graph,also_view_graph
#./run.sh G-LF-AmazonTitles-1.6M $model_type
