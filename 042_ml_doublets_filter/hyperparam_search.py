import subprocess
import random

lr_values = [0.1, 0.01, 0.001, 0.0001]
drop_values = [0.2, 0.5, 0.8]

ps = []
for drop in drop_values:
    for lr in lr_values:
        wd_opt = '--dropout=' + str(drop)
        lr_opt = '--lr=' + str(lr)
        name_opt = '--name=' + 'doublet_cnn_v1_drop' + str(drop) + '_lr' + str(lr) + '_v' + str(random.randint(0, 10**3))
        command = 'python doublet_model.py ' + wd_opt + ' ' + lr_opt + ' ' +  name_opt
        retcode = subprocess.call(command, shell=True)

