# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 16:08:13 2020

@author: 91960
"""
import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt



DATASET_PATH="F:\speech\dataset1\data_stft.json"

#load data

def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)
    
    inputs= np.array(data["stft"])
    targets=np.array(data["labels"])
    
    return inputs,targets
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
    
    
if __name__ == "__main__":
    #load data
     inputs,targets= load_data(DATASET_PATH)
     #split data into train and test sets
     inputs_train,inputs_test,targets_train,targets_test= train_test_split(inputs,
                                                                           targets,
                                                                           test_size=0.3)
     #building the network architecture
     model = keras.Sequential([
         #input layer
         keras.layers.Flatten(input_shape= (inputs.shape[1],inputs.shape[2])),
         
         #1st hiddend layer
         keras.layers.Dense(512,activation="relu"),
        
         
         #2nd hidden layer
         keras.layers.Dense(256,activation="relu"),
         
         #3rd hidden layer
         keras.layers.Dense(64,activation="relu"),
         
          #output layer
         keras.layers.Dense(10,activation="softmax"),
         
         
         ])
     
     #compile network
     optimizer = keras.optimizers.Adam(learning_rate =0.0001)
     model.compile(optimizer= optimizer,
                   loss ="sparse_categorical_crossentropy",
                   metrics=["accuracy"])
    
     model.summary( )
     
     #train network
     
     history=model.fit(inputs_train,targets_train,
               validation_data=(inputs_test,targets_test),
               epochs=5,
               batch_size=32)
     #file_path=r"F:\Prakash\project\modelnew.h5"
     #model.save(file_path)
     plot_history(history)
     