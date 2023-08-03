# ./run.sh 4,5 8182 G-LF-AmazonTitles-1M also_view
export CUDA_VISIBLE_DEVICES=$1
export MASTER_PORT=$2


work_dir=${HOME}/scratch/XC
dataset_name=$3
graph_type=$4
data_dir=${work_dir}/data/$dataset_name
save_dir=${work_dir}/XC/data/GraphTransoformer/$dataset_name
mkdir -p $save_dir

echo Creating Training dataset.

python convert_xc2gf_dataset.py --trn_x_y=$data_dir/trn_X_Y.txt \
    --graph_trn_x_y=$data_dir/$graph_type'_trn_X_Y.txt' \
    --graph_lbl_x_y=$data_dir/$graph_type'_lbl_X_Y.txt' \
    --trn_raw_text=$data_dir/raw_data/train.raw.txt \
    --lbl_raw_text=$data_dir/raw_data/label.raw.txt \
    --graph_raw_text=$data_dir/raw_data/$graph_type'.raw.txt' \
    --save_path=$save_dir/train.tsv \
    --save_valid --valid_perc=0.3 --valid_path=$save_dir/valid.tsv \

echo Creating Testing dataset.
python convert_xc2gf_dataset.py --trn_x_y=$data_dir/tst_X_Y.txt \
    --graph_trn_x_y=$data_dir/$graph_type'_tst_X_Y.txt' \
    --graph_lbl_x_y=$data_dir/$graph_type'_lbl_X_Y.txt' \
    --trn_raw_text=$data_dir/raw_data/test.raw.txt \
    --lbl_raw_text=$data_dir/raw_data/label.raw.txt \
    --graph_raw_text=$data_dir/raw_data/$graph_type'.raw.txt' \
    --save_path=$save_dir/test.tsv

python main.py --mode="train" \
    --train_data_path=$save_dir/train.tsv \
    --valid_data_path=$save_dir/valid.tsv \
    --test_data_path=$save_dir/test.tsv \
    --savename=GraphFormers_$dataset_name'_'$graph_type \
    --world_size=1 --multi_world_size=1 --token_length=32 --neighbor_num=5 \
    --epochs=100 --log_steps=2 \
    --train_batch_size=50 --valid_batch_size=100 \
    --test_batch_size=100 --embed_batch_size=100 \
    --sm_batch_size=1024 \
    --graph_lbl_x_y=$data_dir/$graph_type'_lbl_X_Y.txt' \
    --lbl_raw_text=$data_dir/raw_data/label.raw.txt \
    --graph_raw_text=$data_dir/raw_data/$graph_type'.raw.txt' \
    --tst_raw_text=$data_dir/raw_data/test.raw.txt \
    --tst_x_y=$data_dir/$graph_type'_tst_X_Y.txt' \
    --lr=1e-6 --top_k=300


python main.py --mode="test_xc" \
   --train_data_path=$save_dir/train.tsv \
   --valid_data_path=$save_dir/valid.tsv \
   --test_data_path=$save_dir/test.tsv \
   --savename=GraphFormers_$dataset_name'_'$graph_type \
   --world_size=3 --multi_world_size=4 --token_length=32 --neighbor_num=5 \
   --epochs=100 --log_steps=2 \
   --train_batch_size=50 --valid_batch_size=100 --test_batch_size=100 --label_batch_size=100 \
   --graph_lbl_x_y=$data_dir/$graph_type'_lbl_X_Y.txt' \
   --lbl_raw_text=$data_dir/raw_data/label.raw.txt \
   --graph_raw_text=$data_dir/raw_data/$graph_type'.raw.txt' \
   --tst_raw_text=$data_dir/raw_data/test.raw.txt \
   --tst_x_y=$data_dir/$graph_type'_tst_X_Y.txt' \
   --load_ckpt_name="" \
   --top_k=3000

