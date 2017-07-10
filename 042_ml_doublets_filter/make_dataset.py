"""
Load all files in a directory (.gz), merge them in a single file
and split them in TRAIN / VAL / TEST data
"""
import gzip
import os
import numpy as np

dir = "data/data/clean"

all_data = []
for fname in os.listdir(dir):
    print("processsing: " + fname)
    with gzip.open(dir + "/" + fname, 'rb') as f:
        data = np.genfromtxt(f, delimiter='\t', dtype=np.float32)
        all_data.append(data)


data = np.vstack(all_data)
np.random.shuffle(data)

rows = data.shape[0]

i_train = rows * 0.5
i_val = rows * 0.75

data_train = data[:i_train, :]
data_val = data[i_train:i_val, :]
data_test = data[i_val:, :]

np.save('data/train.npy', data_train)
np.save('data/val.npy', data_val)
np.save('data/test.npy', data_test)