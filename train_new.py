# -*- coding: utf-8 -*-
"""
Created on Wed Apr 05 16:09:56 2017
@author: Jin Haibo
"""
import time,sys
import numpy
import pp

######参数区
fname=str(sys.argv[1]) if len(sys.argv)>1 else "coffee"
s_parts=4
s_step=5
w_step=5
min_width=5

######函数定义区
def div_list(m,n,step=1):
    sn=[i for i in range(m)]
    w=len(sn)/n if len(sn) % n == 0 else len(sn)/n+1
    return [ sn[w*i : w*(i+1):step] for i in range(n) if len(sn)>w*i] ###列表最后可能有空集产生


def compute(opart,spart,min_width,w_step,data,label):

    best_IOD,best_split,best_sub_ts,best_start,best_width=float("Inf"),0,numpy.array([]),0,0
    ##最终变量，最佳信息增益，分割点，shapelet所在的（子串开始位置，子串宽度）
    #######################################################################################
    for start in spart:
        for width in range(min_width,data.shape[1]-start,w_step):##不同长度的候选子串
        ##子串的不同起始位置
            ###以上循环代码得到不同的候选子串（iobj,width,start）（对象标号，子串宽度，子串开始位置）
            cd=data[opart,start:(start+width)].copy()
            mean=cd.mean(axis=1)
            std=cd.std(axis=1)
            for i in range(cd.shape[0]):
                cd[i,:]=(cd[i,:]-mean[i])/(std[i]+1e-6)
            sub_ts=cd.mean(axis=0)
            sub_ts=(sub_ts-sub_ts.mean())/(sub_ts.std()+1e-6)##归一化候选统计子串

            ##计算一个候选子串跟所有序列对象的距离
            D1,D2=[],[]
            for k in range(data.shape[0]):
                ts_tmp=data[k,start:(start+width)]
                ts_tmp=(ts_tmp-ts_tmp.mean())/(ts_tmp.std()+1e-6)##归一化滑窗子串
                min_dist = numpy.sqrt(numpy.sum(numpy.square(ts_tmp-sub_ts))/width)
                if k >= opart[0] and k <= opart[-1]:
                    D1.append(min_dist)
                else:
                    D2.append(min_dist)
            ##结束计算    一个候选子串跟所有序列对象的距离
            ##开始处理 距离直线，先根据距离排序，再求ID=-(pa*log(pa)+pb*log(pb))
            D1,D2 = numpy.array(D1),numpy.array(D2) 
            mean1,std1,mean2,std2 = D1.mean(), D1.std(), D2.mean(), D2.std()
            IOD = 3 * ( std1 + std2 ) / ( mean2 - mean1 )
            if IOD <= best_IOD and mean2 > mean1 :
                best_IOD, best_sub_ts, best_start, best_width = IOD, sub_ts, start, width
                best_split = mean1+ (mean2-mean1) * std1 / (std1+std2)
                
    
    print "hello",best_start,best_width,best_IOD
    rs_ts = numpy.zeros(data.shape[1])
    rs_ts[best_start:(best_start + best_width)] = best_sub_ts
    return [rs_ts, best_start, best_width, best_split, best_IOD]


######装载数据，区分类标和序列
d=numpy.loadtxt("./data/"+fname+"_train",delimiter=",")
d=d[numpy.argsort(d[:,0])]
cls=d[:,0].astype(int)
data =d[:,1:]
len_obj=data.shape[0]
len_ts=data.shape[1]

######创建任务服务器
##ppservers=("*",)
##job_server=pp.Server(ppservers=ppservers)
job_server=pp.Server()
###
print "begin computing...."
print u"序列个数：",len_obj,u"序列长度：",len_ts
###模型数据定义(类别，起始，长度，分割点，最大增益，数据本身)
model=numpy.zeros((len(set(cls)),len_ts+5))
count=0
######对每一个类别进行循环
cids=set(cls)
for cid in cids:
    label=cls.copy()
    label[cls==cid]=0  ##待考察类
    label[cls!=cid]=1  ##其它类
    jobs,res=[],[]
    objs=numpy.argwhere(label==0).reshape(-1)
    print objs
    starts=div_list( len_ts-min_width,s_parts,s_step )
    
    start_t=time.time()
    #print starts
    for s in starts:
        jobs.append(job_server.submit(compute,(objs,s,min_width,w_step,data,label),(),("numpy",)))
    for job in jobs:
        res.append( job() )
    print res
    max_ind=numpy.argmin(numpy.array([item[1:] for item in res])[:,3])
    best_res=res[max_ind][:]
    
    model[count,0]=cid
    model[count,1:5]=numpy.array(best_res[1:])
    model[count,5:]=best_res[0]
    count+=1
    print job_server.print_stats()
    print time.time()-start_t


model=model[numpy.argsort(model[:,4]),:]
numpy.savetxt("./data/model_"+fname,model,delimiter=",") 
