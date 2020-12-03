# import libraries
import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

from google.cloud import storage
import fire
import subprocess
import pickle
import time


def train(bucket, train_file, test_file, epochs=100, batch_size=10):
    # Read in the data
    train_loc = "gs://{0}/{1}".format(bucket, train_file)
    train_data = pd.read_csv(train_loc, index_col=0)
    X_train = train_data.to_numpy()
    
    test_data = pd.read_csv(test_loc, index_col=0)
    X_test = test_data.to_numpy()
    
    # reshape inputs for LSTM [samples, timesteps, features]
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    print("Training data shape:", X_train.shape)
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    print("Test data shape:", X_test.shape)   
    
    
    # define the model network for the autoencoder
    def autoencoder_model(X):
        inputs = Input(shape=(X.shape[1], X.shape[2]))
        L1 = LSTM(16, activation='relu', return_sequences=True, 
                  kernel_regularizer=regularizers.l2(0.00))(inputs)
        L2 = LSTM(4, activation='relu', return_sequences=False)(L1)
        L3 = RepeatVector(X.shape[1])(L2)
        L4 = LSTM(4, activation='relu', return_sequences=True)(L3)
        L5 = LSTM(16, activation='relu', return_sequences=True)(L4)
        output = TimeDistributed(Dense(X.shape[2]))(L5)    
        model = Model(inputs=inputs, outputs=output)
        return model
    
    
    # create the autoencoder model
    model = autoencoder_model(X_train)
    model.compile(optimizer='adam', loss='mae')
    model.summary()
    
    # fit the model to the data
    history = model.fit(X_train, X_train, epochs=nb_epochs, batch_size=batch_size,
                        validation_split=0.1).history
    
    # Save the model to the job dir and GCS
    model_filename = "model.h5"
    model.save(model_filename)
    gcs_model_path = "{0}/model/{1}".format(bucket, model_filename)
    subprocess.check_call(['gsutil', 'cp', model_filename, gcs_model_path], stderr=sys.stdout)
    
    # Save the metrics to the job dir and GCS
    columns = ['loss', 'val_loss']
    metrics = pd.DataFrame(columns=columns)
    metrics['loss'] = history['loss']
    metrics['val_loss'] = history['val_loss']
    metrics_filename = "metrics.csv"
    metrics.to_csv(metrics_filename, index=False)
    gcs_model_path = "{0}/metrics/{1}".format(bucket, metrics_filename)
    subprocess.check_call(['gsutil', 'cp', model_filename, gcs_model_path], stderr=sys.stdout)
    
    
if __name__ == "__main__":
    fire.Fire(train_evaluate)
