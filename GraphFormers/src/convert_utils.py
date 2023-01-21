import random
import pandas as pd
import numpy as np
from tqdm import tqdm
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


def extract_title_data(filename):

    with open(filename, encoding='utf-8') as file:
        content = file.readlines()

    raw_txt = list()
    for line in content:
        line = line[:-1]
        _, text = line.split('->')
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


def xc_to_graphformer_labels(graph_lbl_x_y, lbl_txt, graph_txt):
    label_and_neighbours_list = []
    for r, row in tqdm(enumerate(graph_lbl_x_y), total=len(lbl_txt)):
        label_and_neighbours = []
        cols = row.indices
        label = lbl_txt[r]
        label_and_neighbours.append(label)
        label_and_neighbours.extend(get_neighbours(r, graph_lbl_x_y,
                                                   graph_txt))

        label_and_neighbours_list.append(label_and_neighbours)

    return label_and_neighbours_list


def save_graphformer_data(filename, query_and_neighbours_list, key_and_neighbours_list):
    with open(filename, mode='w', encoding='utf-8') as file:
        for query_and_neighbours, key_and_neighbours in zip(query_and_neighbours_list, key_and_neighbours_list):
            query_and_neighbours_txt = "|'|".join(query_and_neighbours)
            key_and_neighbours_txt = "|'|".join(key_and_neighbours)
            line = f"{query_and_neighbours_txt}\t{key_and_neighbours_txt}\n"
            file.write(line)


