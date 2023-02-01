import argparse
from convert_utils import *
from sklearn import preprocessing

parser = argparse.ArgumentParser(description='Code for converting XC data to a format that could run ELIAS.')
parser.add_argument('--xc_dir', type=str, help='Directory where is XC dataset is stored.')
parser.add_argument('--data_name', type=str, help='Name of the XC dataset.')
parser.add_argument('--save_dir', type=str, help='Directory to save the convert dataset.')

args = parser.parse_args()

xc_dir = args.xc_dir
save_dir = args.save_dir
xc_dataset_name = args.data_name


"""
Classification matrix
"""

trn_x_y_str = read_data(f'{xc_dir}/{xc_dataset_name}/trn_X_Y.txt')
trn_x_y = extract_xc_data(trn_x_y_str)

tst_x_y_str = read_data(f'{xc_dir}/{xc_dataset_name}/tst_X_Y.txt')
tst_x_y = extract_xc_data(tst_x_y_str)

elias_dir = f'{save_dir}/{xc_dataset_name}'
os.makedirs(elias_dir, exist_ok=True)

sp.save_npz(f'{elias_dir}/Y.trn.npz', trn_x_y)
sp.save_npz(f'{elias_dir}/Y.tst.npz', tst_x_y)



"""
Text data
"""

text_dir = f'{xc_dir}/{xc_dataset_name}/raw_data'
elias_dir = f'{save_dir}/{xc_dataset_name}/raw'
os.makedirs(elias_dir, exist_ok=True)

trn_x_txt = read_data(f'{text_dir}/train.raw.txt')
trn_x = extract_xc_text(trn_x_txt)
with open(f'{elias_dir}/trn_X.txt', 'w') as file:
    file.writelines(trn_x)

tst_x_txt = read_data(f'{text_dir}/test.raw.txt')
tst_x = extract_xc_text(tst_x_txt)
with open(f'{elias_dir}/tst_X.txt', 'w') as file:
    file.writelines(tst_x)

lbl_x_txt = read_data(f'{text_dir}/label.raw.txt')
lbl_x = extract_xc_text(lbl_x_txt)
with open(f'{elias_dir}/lbl_X.txt', 'w') as file:
    file.writelines(lbl_x)


"""
Bag of Words
"""

bow_dir = f'{xc_dir}/{xc_dataset_name}'

trn_file = f'{bow_dir}/train_X_Xf.txt'
if not os.path.exists(trn_file):
    raise Exception(f"ERROR:: {trn_file} does not exist.")
trn_bow_str = read_data(trn_file)
trn_bow = extract_xc_data(trn_bow_str)
trn_bow = preprocessing.normalize(trn_bow, norm='l2')


tst_file = f'{bow_dir}/test_X_Xf.txt'
if not os.path.exists(tst_file):
    raise Exception(f"ERROR:: {tst_file} does not exist.")
tst_bow_str = read_data(tst_file)
tst_bow = extract_xc_data(tst_bow_str)
tst_bow = preprocessing.normalize(tst_bow, norm='l2')


lbl_file = f'{bow_dir}/lbl_X_Xf.txt'
if not os.path.exists(lbl_file):
    raise Exception(f"ERROR:: {lbl_file} does not exist.")
lbl_bow_str = read_data(lbl_file)
lbl_bow = extract_xc_data(lbl_bow_str)
lbl_bow = preprocessing.normalize(lbl_bow, norm='l2')


elias_dir = f'{save_dir}/{xc_dataset_name}'
os.makedirs(elias_dir, exist_ok=True)

sp.save_npz(f'{elias_dir}/X.trn.npz', trn_bow)
sp.save_npz(f'{elias_dir}/X.tst.npz', tst_bow)
sp.save_npz(f'{elias_dir}/X.lbl.npz', lbl_bow)


