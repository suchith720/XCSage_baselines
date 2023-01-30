import os
import random
import pandas as pd
import numpy as np
from tqdm import tqdm, tqdm_notebook
from scipy.sparse import csr_matrix


def read_data(filename):
    with open(filename, encoding='utf-8') as file:
        df = file.readlines()
    return df


def extract_xc_data(content):
    header = content[0]
    num_rows, num_cols = header[:-1].split(" ")
    num_rows = int(num_rows)
    num_cols = int(num_cols)

    indptr = [0]
    indices = []
    data = []
    for line in content[1:]:

        line = line[:-1]
        column_value = line.split(" ")
        for cv in column_value:
            if len(cv):
                col_num, value = cv.split(":")
                col_num = int(col_num)
                value = int(value)

                indices.append(col_num)
                data.append(value)
        indptr.append(len(indices))

    train_x_y_mat = csr_matrix((data, indices, indptr), dtype=int)

    return train_x_y_mat


def xc_kgcl_kg(trn_x_y, rel=0):
    str_repr = ""
    for r, row in enumerate(trn_x_y):
        cols = row.indices
        for c in cols:
            str_repr += f"{r} {rel} {c}\n"
    return str_repr


# def xc_kgcl_classification(trn_x_y):
#     str_repr = ""
#     for r, row in enumerate(trn_x_y):
#         cols = row.indices
#         row_str = str(r)+" "+" ".join(map(str, cols))+"\n"
#         str_repr += row_str
#     return str_repr


def xc_kgcl_classification(trn_x_y, tst_x_y, save_dir, valid_pct=0.3):
    os.makedirs(save_dir, exist_ok=True)

    with open(f'{save_dir}/train.txt', 'w') as file_1, \
            open(f'{save_dir}/valid.txt', 'w') as file_2:
        for r, row in tqdm(enumerate(trn_x_y), total=trn_x_y.shape[0]):
            cols = row.indices
            row_str = str(r)+" "+" ".join(map(str, cols))+"\n"

            if random.uniform(0, 1) > valid_pct:
                file_1.write(row_str)
            else:
                file_2.write(row_str)

    with open(f'{save_dir}/test.txt', 'w') as file:
        for r, row in tqdm(enumerate(tst_x_y), total=tst_x_y.shape[0]):
            cols = row.indices
            row_str = str(r+trn_x_y.shape[0])+" "+\
                " ".join(map(str, cols))+"\n"
            file.write(row_str)


def extract_xc_node_id(filename):
    ids = []
    with open(filename) as file:
        for line in file:
            ids.append(line[:-1].split('->')[0])
    return ids


# def create_knowledge_graph(graph, x_ids, y_ids, vocabulary,
#                            kg_str, relation):
#     for r, row in tqdm(enumerate(graph), total=graph.shape[0]):
#         col = row.indices
#         for c in col:
#             node_a = vocabulary.setdefault(x_ids[r], len(vocabulary))
#             node_b = vocabulary.setdefault(y_ids[c], len(vocabulary))
#             kg_str += f'{node_a} {relation} {node_b}\n'
#     return kg_str


def create_knowledge_graph(graphs, graph_ids, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    #import pdb; pdb.set_trace()

    node_vocab = {}
    with open(f'{save_dir}/kg.txt', 'w') as file:
        for n, (graph, graph_id) in enumerate(zip(graphs, graph_ids)):
            for r, row in tqdm(enumerate(graph), total=graph.shape[0]):
                col = row.indices
                for c in col:
                    node_id = node_vocab.setdefault(graph_id[c], len(node_vocab))
                    row_str = f'{r} 0 {node_id}\n'
                    file.write(row_str)


def read_xc_labelGraph(xc_dir, graph_type="graph"):
    graph_label_file = f"{xc_dir}/{graph_type}_lbl_X_Y.txt"
    graph_lbl_x_y_str = read_data(graph_label_file)
    graph_lbl_x_y_mat = extract_xc_data(graph_lbl_x_y_str)

    return graph_lbl_x_y_mat

