if [ $# -lt 1 ]
then
    echo ./elias.sh "<xc_dir>"
    exit 1
fi

echo "Creating dataset G-LF-WikiSeeAlsoTitles-300K"
#./data.sh $1 G-LF-WikiSeeAlsoTitles-300K
echo "Running code"
./run_benchmark.sh G-LF-WikiSeeAlsoTitles-300K

#echo "Creating dataset G-LF-AmazonTitles-1.6M"
#./data.sh $1 G-LF-AmazonTitles-1.6M
#echo "Running code"
#./run_benchmark.sh G-LF-AmazonTitles-1.6M
