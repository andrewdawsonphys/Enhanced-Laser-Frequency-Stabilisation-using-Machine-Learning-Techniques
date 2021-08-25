import numpy as np
import matplotlib.pyplot as plt
import datetime
import scipy.fftpack
from scipy.interpolate import interp1d
import tensorflow as tf
from tensorflow import keras
import pandas as pd

def dnn_keras_tspred_model(train_data):
    
  model = keras.Sequential([
    keras.layers.Dense(32, activation=tf.nn.relu,
                       input_shape=(train_data.shape[1],)),
    keras.layers.Dense(8, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])
  optimizer = tf.keras.optimizers.Adam()
  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae']) 
  model.summary()
  return model

def trainModel():
     
    df = pd.read_csv("simulatedPIDdata.csv")
    tm = df['time'].to_numpy()
    ypn = df['startingSignal'].to_numpy()
    y = df['finalSignal'].to_numpy()
    
    num_train_data = int(0.75 * len(y))
    num_test_data = len(y) - num_train_data
    
    # plt.plot(ypn,label='Before PID')
    # plt.plot(y,label='After PID')
    # plt.legend()
    
    #prepare the train_data and train_labels
    dnn_numinputs = 64
    num_train_batch = 0
    train_data = []
    for k in range(num_train_data-dnn_numinputs-1):
      train_data = np.concatenate((train_data,ypn[k:k+dnn_numinputs]));
      num_train_batch = num_train_batch + 1  
      
    train_data = np.reshape(train_data, (num_train_batch,dnn_numinputs))
    train_labels = y[dnn_numinputs:num_train_batch+dnn_numinputs]
    
    model = dnn_keras_tspred_model(train_data)
    EPOCHS = 300
    strt_time = datetime.datetime.now()
    history = model.fit(train_data, train_labels, epochs=EPOCHS,
                      validation_split=0.2, verbose=0,
                      callbacks=[])
    curr_time = datetime.datetime.now()
    timedelta = curr_time - strt_time
    dnn_train_time = timedelta.total_seconds()
    print("DNN training done. Time elapsed: ", timedelta.total_seconds(), "s")
    # plt.plot(history.epoch, np.array(history.history['val_loss']),
    #             label = 'Val loss')
    # plt.show()
        
    num_test_batch = 0
    strt_idx = num_train_batch
    test_data=[]
    for k in range(strt_idx, strt_idx+num_test_data-dnn_numinputs-1):
      test_data = np.concatenate((test_data,ypn[k:k+dnn_numinputs]));
      num_test_batch = num_test_batch + 1  
    test_data = np.reshape(test_data, (num_test_batch, dnn_numinputs))
    test_labels = y[strt_idx+dnn_numinputs:strt_idx+num_test_batch+dnn_numinputs]
    
    dnn_predictions = model.predict(test_data).flatten()
    keras_dnn_err = test_labels - dnn_predictions
    print("Score : {}".format(sum(keras_dnn_err)))
    return dnn_predictions,test_labels

# Input raw laser signal to this function, outputs the predicted signal after PID is applied.
def makePrediction(signal):
    print("test")

dnn_predictions,test_labels = trainModel()

plt.plot(dnn_predictions,label='DNN_Predictions')
plt.plot(test_labels,label='test_labels')
plt.legend()





