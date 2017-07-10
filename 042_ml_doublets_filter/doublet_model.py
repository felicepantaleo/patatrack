"""
Doublet model with hit shapes and info features.
"""
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
from keras.constraints import max_norm
from keras.callbacks import EarlyStopping, ModelCheckpoint


DEBUG = True
DEBUG = DEBUG and os.name == 'nt'  # DEBUG always False for server

t_now = '{0:%Y-%m-%d_%H-%M-%S}'.format(datetime.datetime.now())
log_dir = "models/cnn_adriano/run_" + t_now
remote_data = '/eos/cms/store/cmst3/group/dehep/convPixels/clean/' 

train_fname = 'data/train' if not DEBUG else 'data/debug.npy'
val_fname = 'data/val' if not DEBUG else 'data/debug.npy'
test_fname = 'data/test' if not DEBUG else 'data/debug.npy'

train_data = Dataset(train_fname)
val_data = Dataset(val_fname)
test_data = Dataset(test_fname)
X_hit, X_info, y = train_data.get_data()
X_val_hit, X_val_info, y_val = val_data.get_data()
X_test_hit, X_test_info, y_test = test_data.get_data()


# Model configuration
LR = 0.001  # TODO: currently not used (adam optimizer)
PATIENCE = 5
EPOCHS = 300 if not DEBUG else 3
BATCH_SIZE = 32
DROP = 0.5
#print("Model settings: ")

hit_shapes = Input(shape=(8, 8, 2), name='hit_shape_input')
infos = Input(shape=(len(dataset.featurelabs),), name='info_input')

drop = Dropout(DROP)(hit_shapes)
conv = Conv2D(32, (3, 3), activation='relu', padding='same', data_format="channels_last", name='conv1')(drop)
conv = Conv2D(32, (3, 3), activation='relu', padding='same', data_format="channels_last", name='conv2')(conv)
pool = MaxPooling2D(pool_size=(2, 2), padding='same', data_format="channels_last", name='pool1')(conv)

conv = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_last", name='conv3')(pool)
conv = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_last", name='conv4')(conv)
pool = MaxPooling2D(pool_size=(2, 2), padding='same', data_format="channels_last", name='pool2')(conv)

flat = Flatten()(pool)
#info_drop = Dropout(infos)
concat = concatenate([flat, infos])

dense = Dense(64, activation='relu', kernel_constraint=max_norm(2.), name='dense')(concat)
drop = Dropout(DROP)(dense)
pred = Dense(2, activation='softmax', kernel_constraint=max_norm(2.), name='output')(drop)

model = Model(inputs=[hit_shapes, infos], outputs=pred)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

print('Training')
# Save the best model during validation and bail out of training early if we're not improving
_, tmpfn = tempfile.mkstemp()
callbacks = [EarlyStopping(patience=PATIENCE), ModelCheckpoint(tmpfn, save_best_only=True, save_weights_only=True)]
model.fit([X_hit, X_info], y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=([X_val_hit, X_val_info], y_val), 
        callbacks=callbacks)

# TODO: should really restore the checkpoint!
# Restore the best found model during validation
model.load_weights(tmpfn)

loss, acc = model.evaluate([X_test_hit, X_test_info], y_test, batch_size=128)
print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))


fname = "models/cnn_doublet/model_" + t_now
model.save_weights(fname + ".h5", overwrite=True)
with open(fname + ".json", "w") as outfile:
    json.dump(model.to_json(), outfile)

# del model  # deletes the existing model

