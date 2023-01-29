import argparse
from src.convert_utils import *

parser = argparse.ArgumentParser(description='Program to convert data from graphFormer format to XC format')
parser.add_argument('--trn_x_y', type=str, help='Path to trn_x_y.')
parser.add_argument('--graph_trn_x_y', type=str, help='Path to graph_trn_x_y.')
parser.add_argument('--graph_lbl_x_y', type=str, help='Path to graph_lbl_x_y.')
parser.add_argument('--trn_raw_text', type=str, help='Path to trn_raw_text.')
parser.add_argument('--lbl_raw_text', type=str, help='Path to lbl_raw_text.')
parser.add_argument('--graph_raw_text', type=str, help='Path to graph_raw_text.')
parser.add_argument('--save_path', type=str, help='Path to save the converted dataset.')
parser.add_argument('--save_valid', action='store_true', help='Flag to denote if validation set has to be created')
parser.add_argument('--valid_perc', type=float, help='Percentage of data to be converted to validation set', default=0.3)
parser.add_argument('--valid_path', type=str, help='Path to save validation dataset.')



if __name__ == '__main__':
    args = parser.parse_args()

    """
    Read Data.
    """

    train_file = args.trn_x_y
    train_text_file = args.trn_raw_text
    label_text_file = args.lbl_raw_text

    trn_x_y_str = read_data(train_file)
    trn_x_y_mat = extract_xc_data(trn_x_y_str)
    trn_raw_txt = extract_title_data(train_text_file)
    lbl_raw_txt = extract_title_data(label_text_file)

    graph_train_file = args.graph_trn_x_y
    graph_label_file = args.graph_lbl_x_y
    graph_text_file = args.graph_raw_text

    graph_trn_x_y_str = read_data(graph_train_file)
    graph_trn_x_y_mat = extract_xc_data(graph_trn_x_y_str)
    graph_lbl_x_y_str = read_data(graph_label_file)
    graph_lbl_x_y_mat = extract_xc_data(graph_lbl_x_y_str)
    graph_raw_txt = extract_title_data(graph_text_file)


    """
    Convert dataset.
    """
    query_and_neighbours, key_and_neighbours = xc_to_graphformer(trn_x_y_mat,
                                                                graph_trn_x_y_mat,
                                                                graph_lbl_x_y_mat,
                                                                trn_raw_txt,
                                                                lbl_raw_txt,
                                                                graph_raw_txt)

#     query_and_neighbours, key_and_neighbours = xc_to_graphformer_2(trn_x_y_mat,
#                                                                  graph_trn_x_y_mat,
#                                                                  graph_lbl_x_y_mat,
#                                                                  trn_raw_txt,
#                                                                  lbl_raw_txt,
#                                                                  graph_raw_txt)

    """
    Save dataset.
    """
    save_path = args.save_path
    save_valid = args.save_valid
    valid_path = args.valid_path
    valid_perc = args.valid_perc

    if save_valid:
        def get_ele(lol, idx):
            return list(map(lambda x: lol[x], idx))
        
        idx = np.arange(len(query_and_neighbours))
        print(len(idx))
        random.shuffle(idx)
        
        valid_len = int(len(query_and_neighbours)*valid_perc)

        query_and_neighbours_train = get_ele(query_and_neighbours, idx[:-valid_len])
        query_and_neighbours_valid = get_ele(query_and_neighbours, idx[-valid_len:])

        key_and_neighbours_train = get_ele(key_and_neighbours, idx[:-valid_len])
        key_and_neighbours_valid = get_ele(key_and_neighbours, idx[-valid_len:])

        save_graphformer_data(save_path, query_and_neighbours_train, key_and_neighbours_train)
        save_graphformer_data(valid_path, query_and_neighbours_valid, key_and_neighbours_valid)
    else:
        save_graphformer_data(save_path, query_and_neighbours, key_and_neighbours)



