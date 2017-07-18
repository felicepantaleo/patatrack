"""
Load all files in a directory (.gz), merge them in a single file
and split them in TRAIN / VAL / TEST data
"""
import gzip
import os
import numpy as np


COMPRESSED = False
dir = "data/data/clean"

def create_numpy_data(files):
    all_data = []
    for fname in files:
        print("processsing: " + fname)
        with gzip.open(dir + "/" + fname, 'rb') as f:
            data = np.genfromtxt(f, delimiter='\t', dtype=np.float32)
            all_data.append(data)
    data = np.vstack(all_data)
    return data


fnames = os.listdir()
n_files = len(fnames)
i_train = int(n_files * 0.5)
i_val = int(n_files * 0.75)

data_train = create_numpy_data(fnames[:i_train])
data_val = create_numpy_data(fnames[i_train:i_val])
data_test = create_numpy_data(fnames[i_val:])
data_debug = data_train[:500, :]

if COMPRESSED:
    np.savez_compressed('data/train.npz', data_train)
    np.savez_compressed('data/val.npz', data_val)
    np.savez_compressed('data/test.npz', data_test)
    np.savez_compressed('data/debug.npz', data_debug)
else:    
    np.save('data/train.npy', data_train)
    np.save('data/val.npy', data_val)
    np.save('data/test.npy', data_test)
    np.save('data/debug.npy', data_debug)
    