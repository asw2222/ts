# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 10:40:32 2017
@author: jin haibo
"""
import numpy as np
import sys
import matplotlib.pyplot as plt

def normalize(data):
    mean=data.mean(axis=1)
    std=data.std(axis=1)
    for i in range(data.shape[0]):
        data[i,:]=(data[i,:]-mean[i])/std[i]
    return data


fname=str(sys.argv[1]) if len(sys.argv)>1 else "coffee_train"

d=np.loadtxt("./data/"+fname,delimiter=",")
label=d[:,0].astype(int)
data = normalize(d[:,1:])
len_obj=len(label)

plt.figure(figsize=(20,10), dpi=80)

i=1
for cid in set(label):
    plt.subplot(len(set(label)),1,i)
    plt.plot(data[label==cid].T)
    i+=1




















