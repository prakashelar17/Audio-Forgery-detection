# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 04:15:39 2020

@author: 91960
"""
import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras



DATASET_PATH="F:\speech\dataset\data.json"

#load data

def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)
    
    inputs= np.array(data["mfcc"])
    targets=np.array(data["labels"])
    
    return inputs,targets

def Prepare_datasplit(test_size,validation_size):
   #load data
   X,Y = load_data(DATASET_PATH)
    
   #prepare test data
   X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=test_size)
    
   #Prepare train data
   X_train,X_validation,Y_train,Y_validation=train_test_split(X_train,Y_train,test_size=validation_size)
    
   #3D array
   X_train=X_train[...,np.newaxis]
   X_validation=X_validation[...,np.newaxis]
   X_test=X_test[...,np.newaxis]

   return X_train,X_test,X_validation,Y_train,Y_test,Y_validation

def build_model(input_shape):
    
   #create model
   model=keras.Sequential()
    
   # 1st convo  layer
   model.add(keras.layers.Conv2D(128,(3,3),activation='relu',input_shape=input_shape))
   model.add(keras.layers.MaxPool2D((3,3),strides=(2,2),padding='same'))
   model.add(keras.layers.BatchNormalization())
    
    #2nd convo layer
   model.add(keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=input_shape))
   model.add(keras.layers.MaxPool2D((3,3),strides=(2,2),padding='same'))
   model.add(keras.layers.BatchNormalization())
    
   #3nd convo layer
   model.add(keras.layers.Conv2D(32,(2,2),activation='relu',input_shape=input_shape))
   model.add(keras.layers.MaxPool2D((3,3),strides=(2,2),padding='same'))
   model.add(keras.layers.BatchNormalization())
   
   #flatten output layer and feed it to dense layer
   model.add(keras.layers.Flatten())
   model.add(keras.layers.Dense(64,activation='relu'))
   
   #output layer
   model.add(keras.layers.Dense(4,activation='softmax'))
   
   return model

def prediction(model,X,Y):
    
    data_class=['COPY-MOVE', 'INSERTION', 'NOT TAMPERED', 'SLICING']
    X=X[np.newaxis,...]
    
    #prediction
    prediction = model.predict(X)
    
    #extract index with max value
    predicted_index = np.argmax(prediction,axis=1)
    
    print("Expected index: {}, Predicted index:{}.".format(data_class[int(Y)],data_class[int(predicted_index)]))

if __name__ == "__main__":
    
    # prepare train,validation,test data
    X_train,X_test,X_validation,Y_train,Y_test,Y_validation=Prepare_datasplit(0.25,0.2)
    
    # build CNN model
    input_shape=(X_train.shape[1],X_train.shape[2],X_train.shape[3])
    model =build_model(input_shape)
    
    #compile network
    optimizer= keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])
    
    #train CNN
    model.fit(X_train,Y_train,validation_data=(X_validation,Y_validation),batch_size=32,epochs=100)
    
    # evaluate the CNN on test set
    test_error,test_accuracy= model.evaluate(X_test,Y_test,verbose=1)
    print("Accuracy on test set is:{}".format(test_accuracy))
    

    # make prediction on a sample  
    X= X_test[100]
    Y= Y_test[100]
    
    prediction(model,X,Y)
    
    print(model.predict)
     
    
        
    
    
    
    
    
    
    
    
    