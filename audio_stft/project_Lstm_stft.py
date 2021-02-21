# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 04:15:39 2020

@author: 91960
"""
import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import librosa
from keras.models import load_model


DATASET_PATH="F:\speech\dataset\data_stft.json"

#load data

def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)
    
    inputs= np.array(data["stft"])
    targets=np.array(data["labels"])
    
    
    return inputs,targets

def test_data_mfcc(filename,n_fft = 2048,hop_length=512):
      signal,sr=librosa.load(filename)
      stft = librosa.stft(signal,n_fft=n_fft,hop_length=hop_length)
      stft1=stft.T
      return stft1
def plot_history(history):
    
    fig , axs =plt.subplots(2)
    
    #create accuracy subplot
    axs[0].plot(history.history["accuracy"],label="train accuracy")
    axs[0].plot(history.history["val_accuracy"],label="test accuracy")
    axs[0].set_ylabel("accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")
    
    #create error subplot
    axs[1].plot(history.history["loss"],label="train error")
    axs[1].plot(history.history["val_loss"],label="test error")
    axs[1].set_ylabel("error")
    axs[1].set_xlabel("epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("error eval")
    
    plt.show()
def Prepare_datasplit(test_size,validation_size):
   #load data
   X,Y = load_data(DATASET_PATH)
    
   #prepare test data
   X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=test_size)
    
   #Prepare train data
   X_train,X_validation,Y_train,Y_validation=train_test_split(X_train,Y_train,test_size=validation_size)
    
   

   return X_train,X_test,X_validation,Y_train,Y_test,Y_validation

def build_model(input_shape):
    
   #create model
   model=keras.Sequential()
    
   # 1st convo  layer
   model.add(keras.layers.LSTM(256,input_shape=input_shape,return_sequences=True))
   model.add(keras.layers.LSTM(64))
   
    
    #2nd convo layer
   
   #model.add(keras.layers.LSTM(64))
   #3nd convo layer
   
   #model.add(keras.layers.LSTM(64))
   #flatten output layer and feed it to dense layer
   
   #model.add(keras.layers.Dense(64,activation='relu'))
   
   #output layer
   model.add(keras.layers.Dense(4,activation='softmax'))
   
   return model

def prediction(model,filename):
    
    data_class=['COPY_MOVE', 'INSERTION', 'NOT TAMPRED', 'SLICING']
    X=test_data_mfcc(filename)
    X=X[np.newaxis,...]
    #prediction
    prediction = model.predict(X)
    
    #extract index with max value
    predicted_index = np.argmax(prediction,axis=1)
    
    print("Predicted index:{}.".format(data_class[int(predicted_index)]))

if __name__ == "__main__":
    
    # prepare train,validation,test data
    X_train,X_test,X_validation,Y_train,Y_test,Y_validation=Prepare_datasplit(0.25,0.2)
    print(X_train.shape)
    print(X_test.shape)
    # build CNN model
    input_shape=(X_train.shape[1],X_train.shape[2])
    print(X_train.shape[1],X_train.shape[2])
    
    model =build_model(input_shape)
    
    #compile network
    optimizer= keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])
    
    #train CNN
    history=model.fit(X_train,Y_train,validation_data=(X_validation,Y_validation),batch_size=32,epochs=100)
    #file_path=r"F:\Prakash\project\modelnew_rnn2.h5"
    #new_model = load_model(file_path)
    #print(new_model.summary())
    # evaluate the CNN on test set
    test_error,test_accuracy= model.evaluate(X_test,Y_test,verbose=1)
    print("Accuracy on test set is:{}".format(test_accuracy))
    
    #file_path=r"F:\Prakash\project\modelnew_rnn2.h5"
    #model.save(file_path)
    
    
    # make prediction on a sample  
    #filename = r"F:\audio_for_testing\testing_audio_original.wav"
   
   #prediction(model,filename)
    plot_history(history)
    
    
    
    
    
    
    
    