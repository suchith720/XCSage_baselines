# Run this script from the src/ directory in this repository.

main_dir=../../../data/G-LF-WikiSeeAlsoTitles-300K
save_dir=../data/G-LF-WikiSeeAlsoTitles-300K

mkdir -p $save_dir

echo Creating Training and Validation dataset.
python src/convert_xc2gf_dataset.py --trn_x_y=$main_dir/trn_X_Y.txt \
    --graph_trn_x_y=$main_dir/graph_trn_X_Y.txt \
    --graph_lbl_x_y=$main_dir/graph_lbl_X_Y.txt \
    --trn_raw_text=$main_dir/raw_data/train.raw.txt \
    --lbl_raw_text=$main_dir/raw_data/label.raw.txt \
    --graph_raw_text=$main_dir/raw_data/graph.raw.txt \
    --save_path=$save_dir/train.tsv \
    --save_valid --valid_perc=0.3 --valid_path=$save_dir/valid.tsv \

echo Creating Testing dataset.
python src/convert_xc2gf_dataset.py --trn_x_y=$main_dir/tst_X_Y.txt \
    --graph_trn_x_y=$main_dir/graph_tst_X_Y.txt \
    --graph_lbl_x_y=$main_dir/graph_lbl_X_Y.txt \
    --trn_raw_text=$main_dir/raw_data/test.raw.txt \
    --lbl_raw_text=$main_dir/raw_data/label.raw.txt \
    --graph_raw_text=$main_dir/raw_data/graph.raw.txt \
    --save_path=$save_dir/test.tsv
