export CUDA_VISIBLE_DEVICES=3

if [ $# -lt 3 ]
then
    echo ./data.sh '<mode> <main_dir> <dataset_name> <graph_type:optional>'
    exit
fi


mode=$1
if [ $mode != "train" ] && [ $mode != "test" ] && [ $mode != "test_xc" ]
then
    echo ERROR:: Mode shoud be one of the following - train, test, test_xc
    exit
fi

dataset_name=$3
if [ $dataset_name != "G-LF-AmazonTitles-1.6M" ] && [ $dataset_name != "G-LF-WikiSeeAlsoTitles-300K" ]
then
    echo ERROR:: dataset_name should be one of G-LF-AmazonTitles-1.6M or G-LF-WikiSeeAlsoTitles-300K.
    exit
fi

main_dir=$2/$dataset_name
if [ ! -e $main_dir ]
then
    echo ERROR:: Directory does not exist, $main_dir
    exit
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


# dataset_name=G-LF-WikiSeeAlsoTitles-300K
# main_dir=../../data/$dataset_name

python main.py --mode=$mode \
    --train_data_path=full_data/$dataset_name/train.tsv \
    --valid_data_path=full_data/$dataset_name/valid.tsv \
    --test_data_path=full_data/$dataset_name/test.tsv \
    --savename=GraphFormers_$dataset_name'_'$graph_type'_3_' \
    --world_size=1 --multi_world_size=1 --token_length=32 --neighbor_num=5 \
    --epochs=100 --log_steps=2 \
    --train_batch_size=50 --valid_batch_size=100 \
    --test_batch_size=100 --embed_batch_size=100 \
    --sm_batch_size=1024 \
    --graph_lbl_x_y=$main_dir/$graph_type'graph_lbl_X_Y.txt' \
    --lbl_raw_text=$main_dir/raw_data/label.raw.txt \
    --graph_raw_text=$main_dir/raw_data/$graph_type'graph.raw.txt' \
    --tst_raw_text=$main_dir/raw_data/test.raw.txt \
    --tst_x_y=$main_dir/$graph_type'graph_tst_X_Y.txt' \
    --load_ckpt_name=ckpt/GraphFormers_$dataset_name'_'$graph_type'_3_'-epoch-1.pt \
    --lr=1e-6 --top_k=3000


#for epoch in $(seq 1 3)
#do
#    python main.py --mode=$mode \
#        --train_data_path=data/$dataset_name/train.tsv \
#        --valid_data_path=data/$dataset_name/valid.tsv \
#        --test_data_path=data/$dataset_name/test.tsv \
#        --savename=GraphFormers_$dataset_name'_'$graph_type$epoch'_3_' \
#        --world_size=3 --multi_world_size=4 --token_length=32 --neighbor_num=5 \
#        --epochs=100 --log_steps=2 \
#        --train_batch_size=50 --valid_batch_size=100 --test_batch_size=100 --label_batch_size=100 \
#        --graph_lbl_x_y=$main_dir/$graph_type'graph_lbl_X_Y.txt' \
#        --lbl_raw_text=$main_dir/raw_data/label.raw.txt \
#        --graph_raw_text=$main_dir/raw_data/$graph_type'graph.raw.txt' \
#        --tst_raw_text=$main_dir/raw_data/test.raw.txt \
#        --tst_x_y=$main_dir/tst_X_Y.txt \
#        --load_ckpt_name=ckpt/GraphFormers_G-LF-WikiSeeAlsoTitles-300K'_3_'-epoch-$epoch.pt \
#        --top_k=3000
#done

#epoch=0
# python main.py --mode=$mode \
#    --train_data_path=full_data/$dataset_name/train.tsv \
#    --valid_data_path=full_data/$dataset_name/valid.tsv \
#    --test_data_path=full_data/$dataset_name/test.tsv \
#    --savename=GraphFormers_$dataset_name'_'$graph_type$epoch'_' \
#    --world_size=3 --multi_world_size=4 --token_length=32 --neighbor_num=5 \
#    --epochs=100 --log_steps=2 \
#    --train_batch_size=50 --valid_batch_size=100 --test_batch_size=100 --label_batch_size=100 \
#    --graph_lbl_x_y=$main_dir/$graph_type'graph_lbl_X_Y.txt' \
#    --lbl_raw_text=$main_dir/raw_data/label.raw.txt \
#    --graph_raw_text=$main_dir/raw_data/$graph_type'graph.raw.txt' \
#    --tst_raw_text=$main_dir/raw_data/test.raw.txt \
#    --tst_x_y=$main_dir/$graph_type'graph_tst_X_Y.txt' \
#    --load_ckpt_name="" \
#    --top_k=3000

