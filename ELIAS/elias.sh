if [ $# -lt 1 ]
then
    echo ./elias.sh "<xc_dir>"
    exit 1
fi

./data.sh $1 G-LF-WikiSeeAlsoTitles-300K
./run_benchmark.sh G-LF-WikiSeeAlsoTitles-300K

./data.sh $1 G-LF-AmazonTitles-1.6M
./run_benchmark.sh G-LF-AmazonTitles-1.6M
