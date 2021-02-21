# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 14:10:39 2020

@author: 91960
"""
import os
import librosa
import json
import math

DATASET_PATH= 'F:\speech\dataset' 
SAMPLE_RATE = 22050
JSON_PATH='F:\speech\dataset\data_mfcc.json' 


def save_mfcc(dataset_path,json_path,n_mfcc=39,n_fft = 2048,hop_length=512):
    
    #dictionary to store data
    data= {
            "mapping":[],
            "mfcc":[],
            "labels":[]
        }
    num_samples=11000
    expected_num_mfcc_vectors= math.ceil(num_samples/hop_length)
    #loop through all the genres
    for i,( dirpath,dirname,filenames) in enumerate(os.walk(dataset_path)):
        
        if dirpath is not dataset_path:
             
            # save the semantic label
            dirpath_componets = dirpath.split("\\")
            semantic_label=dirpath_componets[-1]
            data["mapping"].append(semantic_label)
            print("\n processing {}".format(semantic_label))
            
            # process files for a specific genre
            
            for f in filenames:
                
                # load audio  file
                file_path=os.path.join(dirpath,f)
                signal,sr=librosa.load(file_path, sr = SAMPLE_RATE)
                mfcc = librosa.feature.mfcc(signal, sr=sr , n_fft=n_fft,n_mfcc=n_mfcc,hop_length=hop_length)
                mfcc=mfcc.T
                #store mfcc if it has expected length
                if len(mfcc) == expected_num_mfcc_vectors:
                    data["mfcc"].append(mfcc.tolist())
                    data["labels"].append(i-1)
                
        with open(json_path, "w") as fp:
            json.dump(data,fp, indent=4)
            
if __name__ == "__main__":
        save_mfcc(DATASET_PATH,JSON_PATH)
                