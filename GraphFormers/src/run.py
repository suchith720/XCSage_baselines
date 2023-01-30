import logging
import os
import json
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from src.data_handler import DatasetForMatching, DatasetForNodeAndNeighbours, \
    DataCollatorForMatching, SingleProcessDataLoader, \
    MultiProcessDataLoader
from src.models.tnlrv3.configuration_tnlrv3 import TuringNLRv3Config
from src.convert_utils import *

from tqdm import tqdm


def setup(rank, args):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
    print(f"Process rank : {rank}")
    torch.cuda.set_device(rank)

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)


def cleanup():
    dist.destroy_process_group()


def load_bert(args):
    config = TuringNLRv3Config.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        output_hidden_states=True)
    if args.model_type == "GraphFormers":
        from src.models.modeling_graphformers import GraphFormersForNeighborPredict
        model = GraphFormersForNeighborPredict(config)
        model.load_state_dict(torch.load(args.model_name_or_path, map_location="cpu")['model_state_dict'], strict=False)
        # model = GraphFormersForNeighborPredict.from_pretrained(args.model_name_or_path, config=config)
    elif args.model_type == "GraphSageMax":
        from src.models.modeling_graphsage import GraphSageMaxForNeighborPredict
        model = GraphSageMaxForNeighborPredict.from_pretrained(args.model_name_or_path, config=config)
    return model



def train(local_rank, args, end, load):
    try:
        if local_rank == 0:
            from src.utils import setuplogging
            setuplogging()
        os.environ["RANK"] = str(local_rank)
        setup(local_rank, args)
        if args.fp16:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        model = load_bert(args)
        logging.info('loading model: {}'.format(args.model_type))
        model = model.cuda()

        if load:
            model.load_state_dict(torch.load(args.load_ckpt_name, map_location="cpu"))
            logging.info('load ckpt:{}'.format(args.load_ckpt_name))

        if args.world_size > 1:
            ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        else:
            ddp_model = model

        optimizer = optim.Adam([{'params': ddp_model.parameters(), 'lr': args.lr}])

        data_collator = DataCollatorForMatching(mlm=args.mlm, neighbor_num=args.neighbor_num,
                                                token_length=args.token_length, random_seed=args.random_seed)

        loss = 0.0
        global_step = 0
        best_acc, best_count = 0.0, 0
        for ep in tqdm(range(args.epochs), total=args.epochs):
            start_time = time.time()
            ddp_model.train()
            dataset = DatasetForMatching(file_path=args.train_data_path)
            if args.world_size > 1:
                end.value = False
                dataloader = MultiProcessDataLoader(dataset,
                                                    batch_size=args.train_batch_size,
                                                    collate_fn=data_collator,
                                                    local_rank=local_rank,
                                                    world_size=args.world_size,
                                                    global_end=end)
            else:
                dataloader = SingleProcessDataLoader(dataset, batch_size=args.train_batch_size,
                                                     collate_fn=data_collator, blocking=True)

            for step, batch in enumerate(dataloader):
                if args.enable_gpu:
                    for k, v in batch.items():
                        if v is not None:
                            batch[k] = v.cuda(non_blocking=True)

                if args.fp16:
                    with autocast():
                        batch_loss = ddp_model(**batch)
                else:
                    batch_loss = ddp_model(**batch)
                loss += batch_loss.item()
                optimizer.zero_grad()
                if args.fp16:
                    scaler.scale(batch_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    batch_loss.backward()
                    optimizer.step()

                global_step += 1

                if local_rank == 0 and global_step % args.log_steps == 0:
                    logging.info(
                        '[{}] cost_time:{} step:{}, lr:{}, train_loss: {:.5f}'.format(
                            local_rank, time.time() - start_time, global_step, optimizer.param_groups[0]['lr'],
                                        loss / args.log_steps))
                    loss = 0.0

                dist.barrier()
            logging.info("train time:{}".format(time.time() - start_time))

            if local_rank == 0:
                ckpt_path = os.path.join(args.model_dir, '{}-epoch-{}.pt'.format(args.savename, ep + 1))
                torch.save(model.state_dict(), ckpt_path)
                logging.info(f"Model saved to {ckpt_path}")

                logging.info("Star validation for epoch-{}".format(ep + 1))
                acc = test_single_process(model, args, "valid")
                logging.info("validation time:{}".format(time.time() - start_time))
                if acc > best_acc:
                    ckpt_path = os.path.join(args.model_dir, '{}-best.pt'.format(args.savename))
                    torch.save(model.state_dict(), ckpt_path)
                    logging.info(f"Model saved to {ckpt_path}")

                    best_acc = acc
                    best_count = 0
                else:
                    best_count += 1
                    if best_count >= 2:
                        start_time = time.time()
                        ckpt_path = os.path.join(args.model_dir, '{}-best.pt'.format(args.savename))
                        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
                        logging.info("Star testing for best")

                        acc = test_single_process(model, args, "test")
                        logging.info("test time:{}".format(time.time() - start_time))
                        exit()
            dist.barrier()

        if local_rank == 0:
            start_time = time.time()
            ckpt_path = os.path.join(args.model_dir, '{}-best.pt'.format(args.savename))
            model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
            logging.info("Star testing for best")
            acc = test_single_process(model, args, "test")
            logging.info("test time:{}".format(time.time() - start_time))
        dist.barrier()
        cleanup()
    except:
        import sys
        import traceback
        error_type, error_value, error_trace = sys.exc_info()
        traceback.print_tb(error_trace)
        logging.info(error_value)



@torch.no_grad()
def test_single_process(model, args, mode):
    assert mode in {"valid", "test"}
    model.eval()

    data_collator = DataCollatorForMatching(mlm=args.mlm, neighbor_num=args.neighbor_num,
                                            token_length=args.token_length, random_seed=args.random_seed)
    if mode == "valid":
        dataset = DatasetForMatching(file_path=args.valid_data_path)
        dataloader = SingleProcessDataLoader(dataset, batch_size=args.valid_batch_size, collate_fn=data_collator)
        batch_size = args.valid_batch_size
    elif mode == "test":
        dataset = DatasetForMatching(file_path=args.test_data_path)
        dataloader = SingleProcessDataLoader(dataset, batch_size=args.test_batch_size, collate_fn=data_collator)
        batch_size = args.test_batch_size

    count = 0
    metrics_total = defaultdict(float)
    for step, batch in enumerate(dataloader):
        if args.enable_gpu:
            for k, v in batch.items():
                if v is not None:
                    batch[k] = v.cuda(non_blocking=True)

        metrics = model.test(**batch)
        for k, v in metrics.items():
            metrics_total[k] += v
        count += 1
    for key in metrics_total:
        metrics_total[key] /= count
        logging.info("mode: {}, {}:{}".format(mode, key, metrics_total[key]))
    model.train()
    return metrics_total['main']


def test(args):
    model = load_bert(args)
    logging.info('loading model: {}'.format(args.model_type))
    model = model.cuda()

    checkpoint = torch.load(args.load_ckpt_name, map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    logging.info('load ckpt:{}'.format(args.load_ckpt_name))

    test_single_process(model, args, "test")


@torch.no_grad()
def get_node_embeddings(model, args, node_and_neighbours):
    model.eval()
    dataset = DatasetForNodeAndNeighbours(node_and_neighbours_list=node_and_neighbours)

    data_collator = DataCollatorForMatching(mlm=args.mlm, neighbor_num=args.neighbor_num,
                                            token_length=args.token_length, random_seed=args.random_seed,
                                            is_label=True)

    dataloader = SingleProcessDataLoader(dataset, batch_size=args.embed_batch_size,
                                         collate_fn=data_collator, drop_last=False,
                                         blocking=True)
    node_embeddings = torch.zeros((len(dataset), model.config.hidden_size), device='cpu')
    for step, batch in enumerate(dataloader):
        if args.enable_gpu:
            for k, v in batch.items():
                if v is not None:
                    batch[k] = v.cuda(non_blocking=True)

        node_embed = model.infer(batch['input_ids_query_and_neighbors_batch'],
                                 batch['attention_mask_query_and_neighbors_batch'],
                                 batch['mask_query_and_neighbors_batch'])

        bs = batch['input_ids_query_and_neighbors_batch'].shape[0]
        node_embeddings[step*bs:(step+1)*bs] = node_embed.cpu()

    model.train()

    return node_embeddings

def test_xc(args):
    model = load_bert(args)
    logging.info('loading model: {}'.format(args.model_type))
    model = model.cuda()

    if (args.load_ckpt_name is not None) and (args.load_ckpt_name != ""):
        checkpoint = torch.load(args.load_ckpt_name, map_location="cpu")
        model.load_state_dict(checkpoint, strict=True)
        logging.info('load ckpt:{}'.format(args.load_ckpt_name))

    #computing label embeddings
    label_and_neighbours = extract_node_and_neighbours(args.graph_lbl_x_y, args.lbl_raw_text, args.graph_raw_text)
    label_embeddings = get_node_embeddings(model, args, label_and_neighbours)
    label_embedding_path = os.path.join(args.model_dir, '{}label_embeddings.pt'.format(args.savename))
    torch.save(label_embeddings, label_embedding_path)
    logging.info(f"Label embeddings saved to {label_embedding_path}")

    #computing test embeddings
    test_and_neighbours = extract_node_and_neighbours(args.tst_x_y, args.tst_raw_text, args.graph_raw_text)
    test_embeddings = get_node_embeddings(model, args, test_and_neighbours)
    test_embeddings_path = os.path.join(args.model_dir, '{}test_embeddings.pt'.format(args.savename))
    torch.save(test_embeddings, test_embeddings_path)
    logging.info(f"Test embeddings saved to {test_embeddings_path}")

    return


    print("Computing Score Matrix..")
    test_dataloader = torch.utils.data.DataLoader(test_embeddings, batch_size=args.sm_batch_size,
                                                  shuffle=False, num_workers=4)

    #TODO: Complete the code for computing the scores.“
    scores, indices = [], []
    classifier = label_embeddings.cuda().T
    for test_embed in tqdm(test_dataloader):
        s = torch.mm(test_embed, classifier)
        score, index = torch.topk(s, k=100)
        scores.append(score.detach().cpu().numpy())
        indices.append(index.detach().cpu().numpy())



