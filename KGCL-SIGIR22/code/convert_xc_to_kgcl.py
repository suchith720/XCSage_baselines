import os
import argparse
from convert_utils import *

parser = argparse.ArgumentParser()

parser.add_argument('--xc_dir',
                    type=str,
                    help='path to the xc repository.')
parser.add_argument('--save_dir',
                    type=str,
                    help='path to save the results.')
parser.add_argument('--graph_type',
                    type=str,
                    help='Types of graphs.')
args = parser.parse_args()


if __name__ == "__main__":
    if not os.path.exists(args.xc_dir):
        raise Exception(f"{args.xc_dir} does not exist.")
    xc_dir = args.xc_dir

    if not os.path.exists(args.save_dir):
        raise Exception("{args.save_dir} does not exist.")
    save_dir = args.save_dir


    ## Reading classification data
    print("Reading classification data.")

    train_file = f"{xc_dir}/trn_X_Y.txt"
    trn_x_y_str = read_data(train_file)
    trn_x_y_mat = extract_xc_data(trn_x_y_str)

    test_file = f"{xc_dir}/tst_X_Y.txt"
    tst_x_y_str = read_data(test_file)
    tst_x_y_mat = extract_xc_data(tst_x_y_str)


    # saving classification data
    print("Saving classification data.")
    xc_kgcl_classification(trn_x_y_mat, tst_x_y_mat, args.save_dir)


    # trn_str_repr = xc_kgcl_classification(trn_x_y_mat)
    # os.makedirs(args.save_dir, exist_ok=True)
    # with open(f'{args.save_dir}/train.txt', 'w') as file:
    #     file.write(trn_str_repr)

    # tst_str_repr = xc_kgcl_classification(tst_x_y_mat)
    # os.makedirs(args.save_dir, exist_ok=True)
    # with open(f"{args.save_dir}/test.txt", 'w') as file:
    #     file.write(tst_str_repr)


    ## Graphs
    # train_id_file = f"{xc_dir}/raw_data/train.raw.txt"
    # train_id = extract_xc_node_id(train_id_file)

    # test_id_file = f"{xc_dir}/raw_data/test.raw.txt"
    # test_id = extract_xc_node_id(test_id_file)

    # label_id_file = f"{xc_dir}/raw_data/label.raw.txt"
    # label_id = extract_xc_node_id(label_id_file)

    # graph_id_file = f"{xc_dir}/raw_data/graph.raw.txt"
    # graph_id = extract_xc_node_id(graph_id_file)

    # category_id_file = f"{xc_dir}/raw_data/category.raw.txt"
    # category_id = extract_xc_node_id(category_id_file)


    ## Reading Wikipedia and Category graph

    print("Reading graph data.")
    graphs, graph_ids = [], []
    graph_types = args.graph_type.split(',')
    for gt in graph_types:
        graph_lbl_x_y_mat = read_xc_labelGraph(args.xc_dir,
                                               graph_type=gt)
        graph_id_file = f"{args.xc_dir}/raw_data/{gt}.raw.txt"
        graph_id = extract_xc_node_id(graph_id_file)

        graphs.append(graph_lbl_x_y_mat)
        graph_ids.append(graph_id)


    # creating knowledge graph

    print("Saving graph data.")
    create_knowledge_graph(graphs, graph_ids, args.save_dir)

