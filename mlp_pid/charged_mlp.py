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

file = data_name + "Training_LE_sorted_charged.hdf5"
filename = base_path + file
train = pd.read_hdf(filename, 'event1')

file = data_name + "Val_LE_sorted_charged.hdf5"
filename = base_path + file
val = pd.read_hdf(filename, 'event1')

trainx, trainy = np.array(train.drop('ptype', axis=1)), np.array(train['ptype'])
valx, valy = np.array(val.drop(['ptype', 'group', 'true ptype'], axis=1)), np.array(val['true ptype'])
val_group = np.array(val['group']).astype(np.int64)

### Replace particle tags with integers

# 6:r'$\pi^{+}$', 7:r'$\pi^{-}$', 8:r'$\mu^{-}$', 9:r'$\mu^{+}$'
ptypes = np.array([2212, 321, 211, -13, -11, -2212, -321, -211, 13, 11])
### pi+/- and mu+/- as the same label
ptype = {2212:0, -2212:1, 321:2, -321:3, 11:4, -11:5, 211:6, -211:7, 13:7, -13:6}

trainy = np.array([ptype[trainy[i]] for i in range(len(trainy))])
valy = np.array([ptype[valy[i]] for i in range(len(valy))])

### Define a costum validation loss function

def val_hypothesis_matched(pred_char, valy, val_group):

    confidence_cut = 0.4
    ### Classify particles for each event using highest confidence
    groups, true_group_ind = np.unique(val_group, return_index=True)
    true_ptype_char = valy[true_group_ind]

    pred_char_event = np.maximum.reduceat(pred_char, np.unique(val_group, return_index=True)[1])
    pred_ind_char = np.argmax(np.maximum.reduceat(pred_char, np.unique(val_group, return_index=True)[1]), axis=1)
    max_pred_char = pred_char_event[np.arange(len(pred_ind_char)), pred_ind_char]

    pred_ptype_char = np.argmax(np.maximum.reduceat(pred_char, np.unique(val_group, return_index=True)[1]), axis=1)
    #pred_ptype_char[max_pred_char < confidence_cut] = 13

    events = true_ptype_char.shape[0]
    correct = np.where(true_ptype_char == pred_ptype_char)[0].shape[0]

    return correct / events

### Class for event level classification

class EventLevelValAcc(tf.keras.callbacks.Callback):
    def __init__(self, x_val, y_val, group_val):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.group_val = group_val

    def on_epoch_end(self, epoch, logs=None):

        pred_char = self.model.predict(self.x_val, verbose=0)

        ev_acc = val_hypothesis_matched(pred_char, self.y_val, self.group_val)

        logs = logs or {}
        logs["val_event_acc"] = ev_acc

        print(f" — val_event_acc: {ev_acc:.4f}")

### Define callback

ev_callback = EventLevelValAcc(valx, valy, val_group)

### Convert data into Tensorflow data objects

print(trainx.shape, trainy.shape, valx.shape, valy.shape)

tf_train = tf.data.Dataset.from_tensor_slices((trainx, trainy)).cache()
tf_val = tf.data.Dataset.from_tensor_slices((valx, valy)).cache()

tf_train = tf_train.shuffle(len(tf_train))

tf_train = tf_train.batch(128)
tf_val = tf_val.batch(128)

tf_train = tf_train.prefetch(tf.data.AUTOTUNE)
tf_val = tf_val.prefetch(tf.data.AUTOTUNE)

# Make a model for charged particles

def model_func(hp):
    model = tf.keras.models.Sequential()

    for i in range(1, hp.Int(f"layers", min_value=1, max_value=3)):
        model.add(tf.keras.layers.Dense(units=hp.Int(f"neurons_{i}", min_value=25, max_value=400), activation='relu'))
    
    #model.add(tf.keras.layers.Dense(units=hp.Int(f"neurons", min_value=50, max_value=1000), activation='relu'))
    
    model.add(tf.keras.layers.Dense(len(ptype) - 2, activation = 'softmax'))

    lr = hp.Float(f'learning rate', min_value=10**-4, max_value=10**-2, sampling="LOG")

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=lr), 
              loss = tf.keras.losses.SparseCategoricalCrossentropy(),)
    
    return model

epochs = 50

### Define optimization builder and callback 

tuner = kt.Hyperband(model_func, 
                     objective=kt.Objective("val_event_acc", direction="max"),
                     max_epochs = epochs, 
                     factor=3,
                     hyperband_iterations=1, 
                     directory='charged_model_dir', 
                     project_name='intro_to_kt')

early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_event_acc", mode="max", min_delta=0.001, patience=5)

### Optimize hyperparameters

tuner.search(tf_train, 
             epochs= epochs, 
             validation_data=tf_val, 
             callbacks= [ev_callback, early_stop],
             verbose = 1)

best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

### Train model

model = tuner.hypermodel.build(best_hps)
model.fit(tf_train, 
          epochs= epochs, 
          validation_data=tf_val, 
          callbacks= [ev_callback, early_stop],
          verbose = 1)

### Save Model

suffix = '_paper'
model.save('/projects/mccleary_group/habjan.e/PID_code/Main_analysis/NN_models/Charged_model' + suffix + '.keras')