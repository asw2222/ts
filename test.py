# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 10:30:53 2017

@author: Administrator
"""
import numpy as np
import sys

fname=str(sys.argv[1]) if len(sys.argv)>1 else "coffee"
d=np.loadtxt("./data/"+fname+"_test",delimiter=",")
cls=d[:,0].astype(int)
data =d[:,1:]

model=np.loadtxt("./data/model_"+fname,delimiter=",")  ###模型数据定义(类别，起始，长度，分割点，最大增益，数据本身)

correct=[]

for i in range(data.shape[0]):
    for j in range(model.shape[0]-1):##不考虑最后一个类
        ##shaplet位置 起始+5：起始+5+长度
        shapelet=model[j,model[j,1].astype(int)+5:model[j,1].astype(int)+5+model[j,2].astype(int)]
        shapelet=(shapelet-shapelet.mean())/shapelet.std()
        split=model[j,3]
        label=model[j,0].astype(int)
        
        
        min_dist=float("inf")    
        for start in range(0,len(data[i])-model[j,2].astype(int)):
            sub_ts=data[i,start:start+model[j,2].astype(int)]
            sub_ts=(sub_ts-sub_ts.mean())/sub_ts.std()
            dist=np.sqrt(np.sum(np.square(shapelet-sub_ts))/model[j,2])
            if dist<min_dist:
                min_dist=dist
        
        print label,cls[i],split,min_dist
        if min_dist<=split:
            if cls[i]==label:
                correct.append(1)
                break
            else:
                correct.append(0)
                break
        else:
            if j == (model.shape[0]-1-1):
                if cls[i]==model[-1,0].astype(int):
                    correct.append(1)
                else:
                    correct.append(0)
                
                  
print len(correct),correct,float(sum(correct))/len(correct)
        
        














