import world
import dataloader
import model
import utils
from pprint import pprint
from os.path import join
import os

if world.dataset in ['movielens', 'last-fm', 'MIND', 'yelp2018', 'amazon-book', 'G-LF-WikiSeeAlsoTitles-300K', 'G-LF-AmazonTitles-1.6M']:
    dataset = dataloader.Loader(path=join(world.DATA_PATH,world.dataset))
    kg_dataset = dataloader.KGDataset()
elif world.dataset == 'lastfm':
    dataset = dataloader.LastFM()

print('===========config================')
print(f"PID: {os.getpid()}")
print("KGCN:{}, TransR:{}, N:{}".format(world.kgcn, world.train_trans, world.entity_num_per_item))
print("KGC: {} @ d_prob:{} @ joint:{} @ from_pretrain:{}".format(world.kgc_enable, world.kg_p_drop, world.kgc_joint, world.use_kgc_pretrain))
print("UIC: {} @ d_prob:{} @ temp:{} @ reg:{}".format(world.uicontrast, world.ui_p_drop, world.kgc_temp, world.ssl_reg))
pprint(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print('===========end===================')

MODELS = {
    'lgn': model.KGCL,
    'kgcl': model.KGCL,
    'sgl': model.KGCL,
}
