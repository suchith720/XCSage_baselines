import os
import json
from pathlib import Path

import torch.multiprocessing as mp

from src.parameters import parse_args
from src.run import train, test, test_xc, read_label_and_neighbours, get_test_embeddings
from src.utils import setuplogging

if __name__ == "__main__":

    setuplogging()
    #gpus = ','.join([str(_ + 1) for _ in range(1)])
    #os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12394'

    device_list = ""
    for i, device_id in enumerate(os.environ['CUDA_VISIBLE_DEVICES'].split(',')):
        if i == 0:
            device_list += '"'+device_id+'"'
        else:
            device_list += ',"'+device_id+'"'
    os.environ["GPU_DEVICE_IDS"] = "["+device_list+"]"


    args = parse_args()
    print(os.getcwd())
    #args.log_steps = 5
    #args.world_size = 1  # GPU number
    #args.mode = 'train'
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    cont = True

    #import pdb; pdb.set_trace()
    if args.mode == 'train':
        print('-----------train------------')
        if args.multi_world_size > 1:
            mp.freeze_support()
            mgr = mp.Manager()
            end = mgr.Value('b', False)
            mp.spawn(train,
                     args=(args, end, cont),
                     nprocs=args.multi_world_size,
                     join=True)
        else:
            end = None
            train(0, args, end, cont)

    if args.mode == 'test':
        #args.load_ckpt_name = "/data/workspace/Share/junhan/TopoGram_ckpt/dblp/topogram-pretrain-finetune-dblp-best3.pt"
        print('-------------test--------------')
        test(args)

    if args.mode == 'test_xc':
        #args.load_ckpt_name = "/data/workspace/Share/junhan/TopoGram_ckpt/dblp/topogram-pretrain-finetune-dblp-best3.pt"
        print('-------------test_xc--------------')
        test_xc(args)
