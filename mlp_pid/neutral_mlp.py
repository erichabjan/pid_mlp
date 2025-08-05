import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

import tensorflow as tf
import tensorflow_datasets as tfds
import keras_tuner as kt

from sklearn.metrics import accuracy_score, confusion_matrix

import seaborn as sns

### Import data

dataset_choice = 1
dataset_dic = {1:'pure', 2:'single', 3:'multi'}
data_name = dataset_dic[dataset_choice]

base_path = '/projects/mccleary_group/habjan.e/PID_code/data_processed/'

file = data_name + "Training_LE_sorted_neutral.hdf5"
filename = base_path + file
train = pd.read_hdf(filename, 'event1')

file = data_name + "Val_LE_sorted_neutral.hdf5"
filename = base_path + file
val = pd.read_hdf(filename, 'event1')

trainx, trainy = np.array(train.drop('ptype', axis=1)), np.array(train['ptype']).astype(np.int64)
valx, valy = np.array(val.drop(['ptype', 'group', 'true ptype'], axis=1)), np.array(val['true ptype']).astype(np.int64)
val_group = np.array(val['group']).astype(np.int64)

### Replace particle tags with integers

ptype = {22:0, 130:1, 2112:2}

trainy = np.array([ptype[trainy[i]] for i in range(len(trainy))])
valy = np.array([ptype[valy[i]] for i in range(len(valy))])

### Define a costum validation loss function

def val_hypothesis_matched(pred_neut, valy, val_group):

    confidence_cut = 0.4
    ### Classify particles for each event using highest confidence
    groups, true_group_ind = np.unique(val_group, return_index=True)
    true_ptype_neut = valy[true_group_ind]

    pred_neut_event = np.maximum.reduceat(pred_neut, np.unique(val_group, return_index=True)[1])
    pred_ind_neut = np.argmax(np.maximum.reduceat(pred_neut, np.unique(val_group, return_index=True)[1]), axis=1)
    max_pred_neut = pred_neut_event[np.arange(len(pred_ind_neut)), pred_ind_neut]

    pred_ptype_neut = np.argmax(np.maximum.reduceat(pred_neut, np.unique(val_group, return_index=True)[1]), axis=1)
    pred_ptype_neut[max_pred_neut < confidence_cut] = 13

    events = true_ptype_neut.shape[0]
    correct = np.where(true_ptype_neut == pred_ptype_neut)[0].shape[0]

    return correct / events

### Class for event level classification

class EventLevelValAcc(tf.keras.callbacks.Callback):
    def __init__(self, x_val, y_val, group_val):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.group_val = group_val

    def on_epoch_end(self, epoch, logs=None):

        pred_neut = self.model.predict(self.x_val, verbose=0)

        ev_acc = val_hypothesis_matched(pred_neut, self.y_val, self.group_val)

        logs = logs or {}
        logs["val_event_acc"] = ev_acc

        print(f" — val_event_acc: {ev_acc:.4f}")

### Define callback

ev_callback = EventLevelValAcc(valx, valy, val_group)

### Convert data into TensorFlow objects

tf_train = tf.data.Dataset.from_tensor_slices((trainx, trainy)).cache()
tf_val = tf.data.Dataset.from_tensor_slices((valx, valy)).cache()

tf_train = tf_train.shuffle(len(tf_train))

tf_train = tf_train.batch(128)
tf_val = tf_val.batch(128)

tf_train = tf_train.prefetch(tf.data.AUTOTUNE)
tf_val = tf_val.prefetch(tf.data.AUTOTUNE)

# Make a model for neutral particles

### Create model

def model_func(hp):
    model = tf.keras.models.Sequential()

    #for i in range(1, hp.Int(f"layers", min_value=1, max_value=4)):
     #   model.add(tf.keras.layers.Dense(units=hp.Int(f"neurons_{i}", min_value=100, max_value=600), activation='relu', kernel_regularizer='l1_l2'))
    
    model.add(tf.keras.layers.Dense(units=hp.Int(f"neurons", min_value=50, max_value=1000), activation='relu', kernel_regularizer='l1_l2'))
    
    model.add(tf.keras.layers.Dense(len(ptype), activation = 'softmax'))

    lr = hp.Float(f'learning rate', min_value=10**-4, max_value=10**-2, sampling="LOG")

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=lr), 
                  loss = tf.keras.losses.SparseCategoricalCrossentropy(),)
    
    return model

epochs = 50

### Define tuner and callback

tuner = kt.Hyperband(model_func, 
                     objective=kt.Objective("val_event_acc", direction="max"), 
                     max_epochs = epochs, 
                     factor=3,
                     hyperband_iterations=1, 
                     directory='neutral_model_dir', 
                     project_name='intro_to_kt')

early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_event_acc", mode="max", min_delta=0.001, patience=5)

### Optimize hyperparameters

tuner.search(tf_train, 
             epochs=epochs, 
             validation_data=tf_val, 
             callbacks=[ev_callback, early_stop], 
             verbose = 1)

best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

### Train model

model = tuner.hypermodel.build(best_hps)
model.fit(tf_train, 
          epochs= epochs, 
          validation_data=tf_val, 
          callbacks = [ev_callback, early_stop], 
          verbose = 1)

### Save Model

suffix = '_paper_1_layer'
model.save('/projects/mccleary_group/habjan.e/PID_code/Main_analysis/NN_models/Neutral_model' + suffix + '.keras')