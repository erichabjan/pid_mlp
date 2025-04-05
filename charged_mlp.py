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

base_path = '/Users/erich/Downloads/UConn/Undergraduate-Research/PID_code/data_processed/'

file = data_name + "Training_LE_sorted_charged.hdf5"
filename = base_path + file
train = pd.read_hdf(filename, 'event1')

file = data_name + "Test_LE_sorted_charged.hdf5"
filename = base_path + file
test = pd.read_hdf(filename, 'event1')

trainx, trainy = pd.DataFrame.to_numpy(train.drop('ptype', axis=1)), np.array(train['ptype'])
testx, testy = pd.DataFrame.to_numpy(test.drop(['ptype', 'group', 'true ptype'], axis=1)), np.array(test['true ptype'])

group_test = np.array(test['group'])

### Replace particle tags with integers

ptypes = np.array([np.int64(train['ptype'][i]) for i in range(240000, len(train), 80000)])
ptype = {2212:0, -2212:1, 321:2, -321:3, 11:4, -11:5, 211:6, -211:7, 13:8, -13:9}

trainy = np.array([ptype[trainy[i]] for i in range(len(trainy))])
testy = np.array([ptype[testy[i]] for i in range(len(testy))])

### Convert data into Tensorflow data objects

tf_train = tf.data.Dataset.from_tensor_slices((trainx, trainy)).cache()
tf_test = tf.data.Dataset.from_tensor_slices((testx, testy)).cache()

tf_train = tf_train.shuffle(len(tf_train))

tf_train = tf_train.batch(128)
tf_test = tf_test.batch(128)

tf_train = tf_train.prefetch(tf.data.AUTOTUNE)
tf_test = tf_test.prefetch(tf.data.AUTOTUNE)

# Make a model for charged particles

def model_func(hp):
    model = tf.keras.models.Sequential()

    #for i in range(1, hp.Int(f"layers", min_value=2, max_value=7)):
     #   model.add(tf.keras.layers.Dense(units=hp.Int(f"neurons_{i}", min_value=100, max_value=600), activation='relu', kernel_regularizer='l1_l2'))
    
    model.add(tf.keras.layers.Dense(units=hp.Int(f"neurons", min_value=100, max_value=600), activation='relu', kernel_regularizer='l1_l2'))
    
    model.add(tf.keras.layers.Dense(len(ptype), activation = 'sigmoid'))

    lr = hp.Float(f'learning rate', min_value=10**-4, max_value=10**-2)

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=lr), 
              loss = tf.keras.losses.SparseCategoricalCrossentropy(), 
              metrics = [tf.keras.metrics.SparseCategoricalAccuracy()],)
    
    return model


### Define optimization builder and callback 

tuner = kt.Hyperband(model_func, objective=kt.Objective('val_sparse_categorical_accuracy', direction='max'), 
                     factor=10, directory='charged_model_dir', project_name='intro_to_kt')

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.01, patience=5)

### Run the optimization

tuner.search(tf_train, epochs=50, validation_data=tf_test, callbacks=[callback], verbose = 1)
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

### Train model

model = tuner.hypermodel.build(best_hps)
model.fit(x=trainx, y=trainy, epochs=50, validation_data=(testx, testy), callbacks=[callback], verbose = 1)

### Save Model

model.save('/Users/erich/Downloads/UConn/Undergraduate-Research/PID_code/Main_analysis/NN_models/Charged_model.keras')

