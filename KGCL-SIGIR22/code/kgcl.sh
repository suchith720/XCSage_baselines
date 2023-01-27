if [ $# -lt 1 ]
then
    echo ./kgcl.sh "<main_dir>"
    exit
fi

main_dir=$1
if [ ! -d $main_dir ]
then
    echo Directory does not exist, $main_dir
    exit
fi

#./data.sh $main_dir G-LF-WikiSeeAlsoTitles-300K graph,category
./run.sh G-LF-WikiSeeAlsoTitles-300K

#./data.sh $main_dir G-LF-AmazonTitles-1.6M similar_graph,also_view_graph
#./run.sh G-LF-AmazonTitles-1.6M
