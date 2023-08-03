export CUDA_VISIBLE_DEVICES=$1
export MASTER_PORT=$2


work_dir=${HOME}/XC
dataset_name=$3
graph_type=$4
data_dir=${HOME}/data/$dataset_name
save_dir=${HOME}/XC/data/GraphTransoformer/$dataset_name
mkdir -p $save_dir

python main.py --mode="test_xc" \
   --train_data_path=$save_dir/train.tsv \
   --valid_data_path=$save_dir/valid.tsv \
   --test_data_path=$save_dir/test.tsv \
   --savename=GraphFormers_$dataset_name'_'$graph_type \
   --world_size=3 --multi_world_size=4 --token_length=32 --neighbor_num=5 \
   --epochs=100 --log_steps=2 \
   --train_batch_size=50 --valid_batch_size=100 --test_batch_size=100 \
   --graph_lbl_x_y=$data_dir/$graph_type'_lbl_X_Y.txt' \
   --lbl_raw_text=$data_dir/raw_data/label.raw.txt \
   --graph_raw_text=$data_dir/raw_data/$graph_type'.raw.txt' \
   --tst_raw_text=$data_dir/raw_data/test.raw.txt \
   --tst_x_y=$data_dir/$graph_type'_tst_X_Y.txt' \
   --load_ckpt_name="./ckpt/GraphFormers_${dataset_name}_${graph_type}-epoch-${5}.pt" \
   --top_k=3000

