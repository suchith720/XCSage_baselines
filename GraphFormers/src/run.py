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

from src.data_handler import DatasetForMatching, DatasetForLabels, \
    DataCollatorForMatching, SingleProcessDataLoader, \
    MultiProcessDataLoader
from src.models.tnlrv3.configuration_tnlrv3 import TuringNLRv3Config
from src.convert_utils import *

from tqdm import tqdm


def setup(rank, args):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=args.world_size)

    # device_list = json.loads(os.environ['GPU_DEVICE_IDS'])
    # device_list = list(map(int, device_list))

    print(f"Rank : {rank}")
    # print(f"Device number : {device_list[rank%len(device_list)]}")

    #torch.cuda.set_device(device_list[rank%len(device_list)])
    torch.cuda.set_device(rank)

    # Explicitly setting seed
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


def read_label_and_neighbours(args):
    graph_label_file = args.graph_lbl_x_y
    label_text_file = args.lbl_raw_text
    graph_text_file = args.graph_raw_text

    graph_lbl_x_y_str = read_data(graph_label_file)
    graph_lbl_x_y_mat = extract_xc_data(graph_lbl_x_y_str)

    lbl_raw_txt = extract_title_data(label_text_file)
    graph_raw_txt = extract_title_data(graph_text_file)

    label_and_neighbours = xc_to_graphformer_labels(graph_lbl_x_y_mat,
                                                    lbl_raw_txt,
                                                    graph_raw_txt)

    return label_and_neighbours



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

        label_and_neighbours = read_label_and_neighbours(args)

        # Save label embeddings start
        # label_embeddings = process_labels(model, args, label_and_neighbours)
        # label_embedding_path = os.path.join(args.model_dir, '{}-label_embeddings.pt'.format(args.savename))
        # torch.save(label_embeddings, label_embedding_path)
        # logging.info(f"Label embeddings saved to {label_embedding_path}")
        # Save label embeddings end

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

            # for step, batch in tqdm(enumerate(dataloader), total=int(len(dataset)/args.train_batch_size)):

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

                    # Save label embeddings start
                    label_embeddings = process_labels(model, args, label_and_neighbours)
                    label_embedding_path = os.path.join(args.model_dir, '{}-label_embeddings.pt'.format(args.savename))
                    torch.save(label_embeddings, label_embedding_path)
                    logging.info(f"Label embeddings saved to {label_embedding_path}")
                    # Save label embeddings end

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
def process_labels(model, args, label_and_neighbours):
    dataset = DatasetForLabels(label_and_neighbours_list=label_and_neighbours)

    data_collator = DataCollatorForMatching(mlm=args.mlm, neighbor_num=args.neighbor_num,
                                            token_length=args.token_length, random_seed=args.random_seed,
                                            is_label=True)

    #dataloader = SingleProcessDataLoader(dataset, batch_size=args.label_batch_size, collate_fn=data_collator, blocking=True)
    dataloader = SingleProcessDataLoader(dataset, batch_size=args.label_batch_size, collate_fn=data_collator)
    batch_size = args.label_batch_size

    #import pdb; pdb.set_trace()

    label_embeddings_list = []
    #for step, batch in tqdm(enumerate(dataloader), total=int(len(dataset)/batch_size)):
    for step, batch in enumerate(dataloader):
        if args.enable_gpu:
            for k, v in batch.items():
                if v is not None:
                    batch[k] = v.cuda(non_blocking=True)

        label_embeddings = model.infer(batch['input_ids_query_and_neighbors_batch'],
                                       batch['attention_mask_query_and_neighbors_batch'],
                                       batch['mask_query_and_neighbors_batch'])

        label_embeddings_list.append(label_embeddings.data.cpu())

    label_embeddings = torch.cat(label_embeddings_list, 0)
    model.train()
    return label_embeddings



@torch.no_grad()
def test_single_process(model, args, mode):
    assert mode in {"valid", "test"}
    model.eval()

    data_collator = DataCollatorForMatching(mlm=args.mlm, neighbor_num=args.neighbor_num,
                                            token_length=args.token_length, random_seed=args.random_seed)
    if mode == "valid":
        dataset = DatasetForMatching(file_path=args.valid_data_path)
        #dataloader = SingleProcessDataLoader(dataset, batch_size=args.valid_batch_size, collate_fn=data_collator, blocking=True)
        dataloader = SingleProcessDataLoader(dataset, batch_size=args.valid_batch_size, collate_fn=data_collator)
        batch_size = args.valid_batch_size
    elif mode == "test":
        dataset = DatasetForMatching(file_path=args.test_data_path)
        #dataloader = SingleProcessDataLoader(dataset, batch_size=args.test_batch_size, collate_fn=data_collator, blocking=True)
        dataloader = SingleProcessDataLoader(dataset, batch_size=args.test_batch_size, collate_fn=data_collator)
        batch_size = args.test_batch_size

    count = 0
    metrics_total = defaultdict(float)
    #for step, batch in tqdm(enumerate(dataloader), total=int(len(dataset)/batch_size)):
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


def read_test_and_neighbours(args):
    tst_x_y_file = args.tst_x_y
    tst_text_file = args.tst_raw_text
    graph_text_file = args.graph_raw_text

    tst_x_y_str = read_data(tst_x_y_file)
    tst_x_y_mat = extract_xc_data(tst_x_y_str)

    tst_raw_txt = extract_title_data(tst_text_file)
    graph_raw_txt = extract_title_data(graph_text_file)

    test_and_neighbours = xc_to_graphformer_labels(tst_x_y_mat,
                                                    tst_raw_txt,
                                                    graph_raw_txt)

    return test_and_neighbours

def get_test_embeddings(args):
    model = load_bert(args)
    logging.info('loading model: {}'.format(args.model_type))
    model = model.cuda()

    checkpoint = torch.load(args.load_ckpt_name, map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)
    logging.info('load ckpt:{}'.format(args.load_ckpt_name))

    #import pdb; pdb.set_trace()

    # tst_raw_text = extract_title_data(args.tst_raw_text)
    # test_node = []
    # for raw_text in tst_raw_text:
    #     test_node.append([raw_text])
    test_and_neighbours = read_test_and_neighbours(args)

    test_node_embeddings = process_labels(model, args,
                                          test_and_neighbours)
    return test_node_embeddings

def test_xc(args):
    test_node_embeddings = get_test_embeddings(args)
    test_embeddings_path = os.path.join(args.model_dir, '{}test_embeddings.pt'.format(args.savename))
    torch.save(test_node_embeddings, test_embeddings_path)

    label_embedding_path = os.path.join(args.model_dir, '{}label_embeddings.pt'.format(args.savename))
    label_embeddings = torch.load(label_embedding_path, map_location="cpu")

    test_node_embeddings = test_node_embeddings.cuda()

    batch_size = args.test_batch_size
    score_mat_list = []

    print("Computing Score Matrix..")
    for i in tqdm(range(0, label_embeddings.shape[0], batch_size), total=int(label_embeddings.shape[0]/batch_size)):
    #for i in range(0, label_embeddings.shape[0], batch_size):
        label_embed = label_embeddings[i:i+batch_size].cuda()
        score_mat = test_node_embeddings@label_embed.t()
        score_mat = torch.argsort(score_mat, dim=1, descending=True)[:, :args.top_k]
        score_mat_list.append(score_mat.cpu())

    #import pdb; pdb.set_trace()
    score_mat = torch.cat(score_mat_list, 1)

    score_mat_path = os.path.join(args.model_dir, '{}score_mat.pt'.format(args.savename))
    torch.save(score_mat, score_mat_path)
    logging.info('Saving Score Matrix.')


