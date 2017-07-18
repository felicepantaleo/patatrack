'''
    Hyperparameter selection for NN with RandomizedSearch.
'''
import argparse
import numpy as np
import math
from scipy.stats import randint as sp_randint
import random
import subprocess


parser = argparse.ArgumentParser()
parser.add_argument('--proc', type=int, default=1)
args = parser.parse_args()


log_dir = input('experiment subdirectory:')
log_dir_opt = '--log_dir=models/' + log_dir


"""
This dictionary represent the distribution for the random parameter selection.
The keys are the model's parameter name and the values represent the distribution.
* elements from lists are drawn uniformly at random
* sp_randint(min, max) generate an element in the range [min, max] uniformly
"""
dists = {
    'batch_size': [32, 64, 128],
    'n_epochs': [100, 300, 500],
    'dropout':  [.1, .2, .3, .4, .5, .6, .7, .8, .9],
    'lr': [0.1, 0.01, 0.001, 0.0001, 0.00001],
    'momentum': [0, .1, .5, .9],
    'maxnorm': [1., 2., 5., 20.]
}

ps = []
m = 1
# try different models until the GPU blows up
while True:
    print("---------------------------------------")
    print("Iteration number " + str(m))
    print("---------------------------------------")    
    m += 1

    d = {}
    for k in dists:
        if hasattr(dists[k], '__len__'):
            idx = random.randint(0, len(dists[k]) - 1)
            d[k] = dists[k][idx]
        else:
            d[k] = dists[k].rvs()

    opt_str = ""
    opt_name = "double_cnn_v1"
    for k in d:
        opt_str += " --" + k + "=" + str(d[k])
        opt_name += "_" + k + str(d[k])

    r = str(random.randint(0, 10**3))    
    opt_name += '_v' + r    
    opt_str += " --name=" + opt_name
    opt_str += " --verbose=0"
    
    command = 'python doublet_model.py ' + opt_str + ' ' + log_dir_opt
    p = subprocess.Popen(command, shell=True)

    ps.append(p)
    if len(ps) >= args.proc:
        [p.wait() for p in ps]
        ps = []

