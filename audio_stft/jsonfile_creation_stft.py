# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 08:15:19 2021

@author: 91960
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 14:10:39 2020

@author: 91960
"""
import os
import librosa
import json
import math
import numpy as np

DATASET_PATH= 'F:\speech\dataset1' 
SAMPLE_RATE = 22050
JSON_PATH='F:\speech\dataset1\data_stft.json' 

def save_mfcc(dataset_path,json_path,n_fft = 2048,hop_length=512):
    
    #dictionary to store data
    data= {
            "mapping":[],
            "stft":[],
            "labels":[]
        }
    num_samples=11000
    #expected_num_mfcc_vectors= math.ceil(num_samples/hop_length)
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
                stft = librosa.stft(signal , n_fft=n_fft,hop_length=hop_length)
                stft1 = np.abs(stft)**2
                #print(stft1.shape)
                stft2=stft1.T
                # print(stft2.shape)
                
                #store mfcc if it has expected length
                #if len(mfcc) == expected_num_mfcc_vectors:
                data["stft"].append(stft2.tolist())
                data["labels"].append(i-1)
                
        with open(json_path, "w") as fp:
            json.dump(data,fp, indent=4)
            
if __name__ == "__main__":
        save_mfcc(DATASET_PATH,JSON_PATH)
                