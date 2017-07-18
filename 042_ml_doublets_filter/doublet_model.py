"""
Doublet model with hit shapes and info features.
"""
import socket

if socket.gethostname() == 'cmg-gpu1080':
    print('locking only one GPU.')
    import setGPU

import argparse
import keras
import dataset
import datetime
import json
import tempfile
import os
import numpy as np
from dataset import Dataset
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate, Reshape, Dropout
from keras.models import Model
from keras.utils.np_utils import to_categorical
from keras import optimizers
from keras.constraints import max_norm
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard


DEBUG = os.name == 'nt'  # DEBUG on laptop
if DEBUG:
    print("DEBUG mode")

t_now = '{0:%Y-%m-%d_%H-%M-%S}'.format(datetime.datetime.now())
# Model configuration
parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=10 if not DEBUG else 3,
                    help='number of epochs')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--patience', type=int, default=15)
parser.add_argument('--log_dir', type=str, default="models/cnn_doublet")
parser.add_argument('--name', type=str, default='model_' + t_now)
parser.add_argument('--maxnorm', type=float, default=2.)
parser.add_argument('--verbose', type=int, default=1)
args = parser.parse_args()

log_dir_tf = args.log_dir + '/' + args.name
remote_data = '/eos/cms/store/cmst3/group/dehep/convPixels/clean/' 

print("Loading data...")
data_dir = 'data/'
train_fname = data_dir + 'train.npy' if not DEBUG else 'data/debug.npy'
val_fname = data_dir + 'val.npy' if not DEBUG else 'data/debug.npy'
test_fname = data_dir + 'test.npy' if not DEBUG else 'data/debug.npy'

train_data = Dataset(train_fname).filter('isFlippedIn', 1.0).filter('isFlippedOut', 1.0)
val_data = Dataset(val_fname).filter('isFlippedIn', 1.0).filter('isFlippedOut', 1.0)
test_data = Dataset(test_fname).filter('isFlippedIn', 1.0).filter('isFlippedOut', 1.0)
X_hit, X_info, y = train_data.get_data()
X_val_hit, X_val_info, y_val = val_data.get_data()
X_test_hit, X_test_info, y_test = test_data.get_data()


hit_shapes = Input(shape=(8, 8, 2), name='hit_shape_input')
infos = Input(shape=(len(dataset.featurelabs),), name='info_input')

drop = Dropout(args.dropout)(hit_shapes)
conv = Conv2D(32, (3, 3), activation='relu', padding='same', data_format="channels_last", name='conv1')(drop)
conv = Conv2D(32, (3, 3), activation='relu', padding='same', data_format="channels_last", name='conv2')(conv)
pool = MaxPooling2D(pool_size=(2, 2), padding='same', data_format="channels_last", name='pool1')(conv)

conv = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_last", name='conv3')(pool)
conv = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_last", name='conv4')(conv)
pool = MaxPooling2D(pool_size=(2, 2), padding='same', data_format="channels_last", name='pool2')(conv)


flat = Flatten()(pool)
#info_drop = Dropout(infos)
concat = concatenate([flat, infos])

dense = Dense(64, activation='relu', kernel_constraint=max_norm(args.maxnorm), name='dense')(concat)
drop = Dropout(args.dropout)(dense)
pred = Dense(2, activation='softmax', kernel_constraint=max_norm(args.maxnorm), name='output')(drop)

model = Model(inputs=[hit_shapes, infos], outputs=pred)
my_sgd = optimizers.SGD(lr=args.lr, decay=1e-6, momentum=args.momentum, nesterov=True)
model.compile(optimizer=my_sgd, loss='categorical_crossentropy', metrics=['accuracy'])

if args.verbose:
    model.summary()

print('Training')
# Save the best model during validation and bail out of training early if we're not improving
_, tmpfn = tempfile.mkstemp()
callbacks = [
    EarlyStopping(monitor='val_loss', patience=args.patience), 
    ModelCheckpoint(tmpfn, save_best_only=True, save_weights_only=True), 
    TensorBoard(log_dir=log_dir_tf, histogram_freq=0, write_graph=True, write_images=True)
]
model.fit([X_hit, X_info], y, batch_size=args.batch_size, epochs=args.n_epochs, 
    validation_data=([X_val_hit, X_val_info], y_val), callbacks=callbacks, verbose=args.verbose)

# Restore the best found model during validation
model.load_weights(tmpfn)

loss, acc = model.evaluate([X_test_hit, X_test_info], y_test, batch_size=args.batch_size)
print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))


fname = args.log_dir + "/" + args.name 
print("saving model " + fname)
model.save_weights(fname + ".h5", overwrite=True)
with open(fname + ".json", "w") as outfile:
    json.dump(model.to_json(), outfile)

