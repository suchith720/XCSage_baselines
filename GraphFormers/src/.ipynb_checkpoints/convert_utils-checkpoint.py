import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix
from xclib.data import data_utils as du


def read_data(filename):
    with open(filename, encoding='utf-8') as file:
        df = file.readlines()
    return df


def extract_xc_data(content):
    
    return du.read_sparse_file(content)


def extract_title_data(filename):

    with open(filename, encoding='utf-8') as file:
        content = file.readlines()

    raw_txt = list()
    for line in content:
        line = line.strip()
        _, text = line.split('->', 1)
        raw_txt.append(text)
    return raw_txt


def get_neighbours(row_num, graph, graph_text):
    neighbours_text = []

    neighbours_idx = graph[row_num].indices
    neighbours_idx = neighbours_idx[np.argsort(graph[row_num].data)][::-1]

    for n in neighbours_idx:
        neighbours_text.append(graph_text[n])

    #random.shuffle(neighbours_text)
    return neighbours_text


def xc_to_graphformer(trn_x_y, graph_trn_x_y, graph_lbl_x_y, trn_txt, lbl_txt, graph_txt):
    query_and_neighbours_list, key_and_neighbours_list = [], []

    for r, row in tqdm(enumerate(trn_x_y), total=trn_x_y.shape[0]):
        query_and_neighbours = []
        cols = row.indices

        query = trn_txt[r]
        query_and_neighbours.append(query)
        query_and_neighbours.extend(get_neighbours(r, graph_trn_x_y, graph_txt))

        for c in cols:
            key_and_neighbours = []

            key = lbl_txt[c]
            key_and_neighbours.append(key)
            key_and_neighbours.extend(get_neighbours(c, graph_lbl_x_y, graph_txt))

            query_and_neighbours_list.append(query_and_neighbours)
            key_and_neighbours_list.append(key_and_neighbours)

    return query_and_neighbours_list, key_and_neighbours_list


def xc_to_graphformer_2(trn_x_y, graph_trn_x_y, graph_lbl_x_y, trn_txt, lbl_txt, graph_txt):
    query_and_neighbours_list, key_and_neighbours_list = [], []

    for c in tqdm(range(trn_x_y.shape[1]), total=trn_x_y.shape[1]):
        #import pdb; pdb.set_trace()
        query_and_neighbours, key_and_neighbours = [], []
        col = trn_x_y[:, c].tocsc()
        rows = col.indices

        key = lbl_txt[c]
        key_and_neighbours.append(key)
        key_and_neighbours.extend(get_neighbours(c, graph_lbl_x_y, graph_txt))

        r = np.random.choice(rows)
        query = trn_txt[r]
        query_and_neighbours.append(query)
        query_and_neighbours.extend(get_neighbours(r, graph_trn_x_y, graph_txt))

        query_and_neighbours_list.append(query_and_neighbours)
        key_and_neighbours_list.append(key_and_neighbours)

    return query_and_neighbours_list, key_and_neighbours_list


def save_graphformer_data(filename, query_and_neighbours_list, key_and_neighbours_list):
    with open(filename, mode='w', encoding='utf-8') as file:
        for query_and_neighbours, key_and_neighbours in zip(query_and_neighbours_list, key_and_neighbours_list):
            query_and_neighbours_txt = "|'|".join(query_and_neighbours)
            key_and_neighbours_txt = "|'|".join(key_and_neighbours)
            line = f"{query_and_neighbours_txt}\t{key_and_neighbours_txt}\n"
            file.write(line)


def extract_node_and_neighbours(node_graph_file, node_txt_file, neighbour_txt_file):
    node_graph, node_txt, neighbour_txt = read_xc_node_data(node_graph_file,
                                                            node_txt_file,
                                                            neighbour_txt_file)
    return get_node_and_neighbours(node_graph, node_txt, neighbour_txt)


def read_xc_node_data(node_graph_file, node_txt_file, neighbour_txt_file):
    node_graph = extract_xc_data(node_graph_file)

    node_txt = extract_title_data(node_txt_file)
    neighbour_txt = extract_title_data(neighbour_txt_file)
    return node_graph, node_txt, neighbour_txt


def get_node_and_neighbours(node_graph, node_txt, neighbour_txt):
    node_and_neighbours_list = []
    for r in tqdm(range(len(node_txt))):
        node_and_neighbours = []

        label = node_txt[r]
        node_and_neighbours.append(label)
        node_and_neighbours.extend(get_neighbours(r, node_graph,
                                                   neighbour_txt))

        node_and_neighbours_list.append(node_and_neighbours)
    return node_and_neighbours_list


