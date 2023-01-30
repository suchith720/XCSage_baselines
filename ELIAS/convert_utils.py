import os
import scipy.sparse as sp

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
                value = float(value)

                indices.append(col_num)
                data.append(value)
        indptr.append(len(indices))

    train_x_y_mat = sp.csr_matrix((data, indices, indptr), dtype=float)

    return train_x_y_mat


def extract_xc_text(content):
    trn_x = []
    for line in content:
        _, text = line.split('->')
        trn_x.append(text)
    return trn_x


def read_data(filename):
    with open(filename, encoding='utf-8') as file:
        df = file.readlines()
    return df
