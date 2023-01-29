# Run this script from the src/ directory in this repository.
if [ $# -lt 2 ]
then
    echo ./data.sh '<main_dir> <dataset_name> <graph_type:optional> <save_dir:optional>'
    exit
fi

# #parameters
# dataset_name=G-LF-WikiSeeAlsoTitles-300K
# main_dir=../../data/$dataset_name
# save_dir=./data/$dataset_name
# graph_type=""

dataset_name=$2
main_dir=$1/$dataset_name
if [ ! -e $main_dir ]
then
    echo ERROR:: Directory does not exist, $main_dir
    exit
fi

if [ $# -gt 2 ]
then
    save_dir=$3/$dataset_name
else
    save_dir=./data/$dataset_name
fi

if [ $# -gt 3 ]
then
    graph_type=$4
    if [ $graph_type != "similar_" ] && [ $graph_type != "also_view_" ]
    then
        echo ERROR:: Graph type should be in : {"similar_", "also_view_"}
        exit
    fi
fi

mkdir -p $save_dir

echo Creating Training and Validation dataset.
python convert_xc2gf_dataset.py --trn_x_y=$main_dir/trn_X_Y.txt \
    --graph_trn_x_y=$main_dir/$graph_type'graph_trn_X_Y.txt' \
    --graph_lbl_x_y=$main_dir/$graph_type'graph_lbl_X_Y.txt' \
    --trn_raw_text=$main_dir/raw_data/train.raw.txt \
    --lbl_raw_text=$main_dir/raw_data/label.raw.txt \
    --graph_raw_text=$main_dir/raw_data/$graph_type'graph.raw.txt' \
    --save_path=$save_dir/train.tsv \
    --save_valid --valid_perc=0.3 --valid_path=$save_dir/valid.tsv \

echo Creating Testing dataset.
python convert_xc2gf_dataset.py --trn_x_y=$main_dir/tst_X_Y.txt \
    --graph_trn_x_y=$main_dir/$graph_type'graph_tst_X_Y.txt' \
    --graph_lbl_x_y=$main_dir/$graph_type'graph_lbl_X_Y.txt' \
    --trn_raw_text=$main_dir/raw_data/test.raw.txt \
    --lbl_raw_text=$main_dir/raw_data/label.raw.txt \
    --graph_raw_text=$main_dir/raw_data/$graph_type'graph.raw.txt' \
    --save_path=$save_dir/test.tsv
