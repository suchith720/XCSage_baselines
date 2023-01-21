# XCSage_baselines

## GraphFormer

+ Download data from https://owncloud.iitd.ac.in/nextcloud/index.php/s/2m7zQomKeNxXAZd
+ This will download GraphFormer.zip, which has two folders GraphFormers/ckpt and GraphFormers/TuringModels
+ Add both of them inside GraphFormer directory.

- cd GraphFormer
- ./graphformer.sh <xc_data_path>



output : GraphFormer/ckpt/GraphFormers_<dataset_name>_<graph_type>_score_mat.pt

## KGCL
- cd KGCL-SIGIR22/code
- ./kgcl.sh <xc_data_path>

output : KGCL-SIGIR22/code/output/output/kgc-<dataset_name>-64.pth.tar
