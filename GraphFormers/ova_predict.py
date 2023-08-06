import torch
import os
import sys
import numpy as np
from xclib.data import data_utils as du
import scipy.sparse as sp
from tqdm.auto import tqdm

dataset=f"{sys.argv[1]}"
graph=sys.argv[2]
cuda = int(sys.argv[3])
topk = int(sys.argv[4])
save_path = f"{os.environ['HOME']}/scratch/XC/score_mats/{dataset}/GraphFormers"
data = f"{os.environ['HOME']}/scratch/XC/data/{dataset}"
tst_y = du.read_sparse_file(f"{data}/tst_X_Y.txt")

tst_emb = torch.load(f"./ckpt/GraphFormers_{dataset}_{graph}test_embeddings.pt")
lbl_emb = torch.load(f"./ckpt/GraphFormers_{dataset}_{graph}label_embeddings.pt")

tst_dl = torch.utils.data.DataLoader(tst_emb, batch_size=1024,
                                     shuffle=False, num_workers=10)
lbl_emb = lbl_emb.T.cuda(cuda)
print(tst_emb.shape, lbl_emb.shape)
score_mat = sp.lil_matrix((tst_emb.shape[0], lbl_emb.shape[1]))
scrs, idxs = [], []
for data in tqdm(tst_dl):
    scores = data.cuda(cuda).mm(lbl_emb)
    scr, idx = torch.topk(scores, k=topk)
    scrs.append(scr.cpu().numpy())
    idxs.append(idx.cpu().numpy())
    
scrs = np.vstack(scrs)
idxs = np.vstack(idxs)
index = np.arange(idxs.shape[0]).reshape(-1, 1)
score_mat[index, idxs] = scrs

os.makedirs(save_path, exist_ok=True)
sp.save_npz(f"{save_path}/score.npz", score_mat.tocsr())