if [ $# -lt 1 ]
then
    echo ./run.sh "<data_name=G-LF-WikiSeeAlsoTitles-300K or G-LF-AmazonTitles-1.6M>"
    exit 1
fi

dataset_name=G-LF-WikiSeeAlsoTitles-300K

python3 -m pecos.xmc.xtransformer.train \
    -t Datasets/$dataset_name/raw/trn_X.txt \
    -x Datasets/$dataset_name/X.trn.npz \
    -y Datasets/$dataset_name/Y.trn.npz \
    -m models/$dataset_name/ \
    -tt Datasets/$dataset_name/raw/tst_X.txt \
    -xt Datasets/$dataset_name/X.tst.npz \
    -yt Datasets/$dataset_name/Y.tst.npz \
    --label-feat-path Datasets/$dataset_name/X.lbl.npz
