# XCSage_baselines

## GraphFormer

Download data from https://owncloud.iitd.ac.in/nextcloud/index.php/s/2m7zQomKeNxXAZd <br>
This will download GraphFormer.zip, which has two folders GraphFormers/ckpt and GraphFormers/TuringModels <br>
Add both of them inside GraphFormer directory.

- cd GraphFormer
- ./graphformer.sh <xc_data_path>



output : GraphFormer/ckpt/GraphFormers_<dataset_name>_<graph_type>_score_mat.pt

## KGCL
- cd KGCL-SIGIR22/code
- ./kgcl.sh <xc_data_path> <model_type=kgcl,lgn,sgl>

output : KGCL-SIGIR22/code/output/output/kgc-<dataset_name>-64.pth.tar


## ELIAS
This code required BOW representation of the datapoint, which you can download from here: [G-LF-WikiSeeAlsoTitles-300K](https://owncloud.iitd.ac.in/nextcloud/index.php/s/YqgxmoQ8tt25445), [G-LF-AmazonTitles-1.6M](https://owncloud.iitd.ac.in/nextcloud/index.php/s/dXbPkT6xCcybANz)

- cd ELIAS/
- ./elias.sh <xc_data_path>

Metrics would be printed on the screen.

