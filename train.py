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
n_parts=10
s_parts=30
s_step=5
w_step=5
min_width=5

######函数定义区
def div_list(m,n,step=1):
    sn=[i for i in range(m)]
    w=len(sn)/n if len(sn) % n == 0 else len(sn)/n+1
    return [ sn[w*i : w*(i+1):step] for i in range(n) if len(sn)>w*i] ###列表最后可能有空集产生

def compute(opart,spart,min_width,w_step,data,label):
    best_gain=-float("Inf")
    best_split=0.
    best_i=-1
    best_width=0
    best_start=0
    ##最终变量，最佳信息增益，分割点，shapelet所在的（对象标号，子串宽度，子串开始位置）
    #######################################################################################
    for iobj in opart:##不同对象
        len_ts=len(data[iobj])
        for start in spart:
            for width in range(min_width,len_ts-start,w_step):##不同长度的候选子串
            ##子串的不同起始位置
                ###以上循环代码得到不同的候选子串（iobj,width,start）（对象标号，子串宽度，子串开始位置）
                sub_ts=data[iobj,start:start+width]
                sub_ts=(sub_ts-sub_ts.mean())/(sub_ts.std()+1e-6)##归一化候选子串
                dist_line=numpy.zeros((data.shape[0],2))


                ##计算一个候选子串跟所有序列对象的距离
                for k in range(data.shape[0]):
                    min_dist=float("Inf")
                    ##m_start_min=(start-30) if (start-30) > 0 else 1
                    ##m_start_max=(start+30) if (start+30+width) <len(data[k]) else len(data[k])-width
                    ##for m_start in range(m_start_min,m_start_max,1):
                    for m_start in range(1,len(data[k])-width+1,1):
                        sub_ts_tmp=data[k,m_start:(m_start+width)]
                        sub_ts_tmp=(sub_ts_tmp-sub_ts_tmp.mean())/(sub_ts_tmp.std()+1e-6)##归一化滑窗子串
                        dist = numpy.sqrt(numpy.sum(numpy.square(sub_ts_tmp-sub_ts))/width)
                        if dist<min_dist:
                            min_dist=dist
                    dist_line[k,:]=[ label[k] , min_dist ]##orderline (类标号,距离)

                ##结束计算    一个候选子串跟所有序列对象的距离
                ##开始处理 距离直线，先根据距离排序，再求ID=-(pa*log(pa)+pb*log(pb))

                dist_line=dist_line[numpy.argsort(dist_line[:,1]),:]##orderline排序
                pA= numpy.sum(dist_line[:,0]>0.5)/float(dist_line.shape[0])+1e-6
                pB=1.0-pA + 1e-6
                ID=-(pA*numpy.log(pA)+pB*numpy.log(pB))
                ##开始循环测试距离直线上每个切分点，得到信息增益，并且保存最大的增益信息
                for n in range(1,dist_line.shape[0]):
                    split=(dist_line[n-1,1]+dist_line[n,1])/2
                    D1=dist_line[:n,:]
                    D2=dist_line[n:,:]
                    fD1=n/float(dist_line.shape[0])
                    fD2=1.0-fD1
                    pA1=numpy.sum(D1[:,0]>0.5)/float(D1.shape[0]) +1e-6
                    pB1=1.0-pA1 + 1e-6
                    ID1=-(pA1*numpy.log(pA1)+pB1*numpy.log(pB1))
                    pA2=numpy.sum(D2[:,0]>0.5)/float(D2.shape[0]) +1e-6
                    pB2=1.0-pA2 + 1e-6
                    ID2=-(pA2*numpy.log(pA2)+pB2*numpy.log(pB2))
                    gain=ID-fD1*ID1-fD2*ID2
                    if gain >= best_gain:
                        best_split=split
                        best_gain=gain
                        best_i=iobj
                        best_width=width
                        best_start=start
                 ##循环结束

    print best_i,best_width,best_start,best_gain
    return [best_i,best_start,best_width,best_split,best_gain]
######装载数据，区分类标和序列
d=numpy.loadtxt("./data/"+fname+"_train",delimiter=",")
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
for cid in set(cls):
    label=cls.copy()
    label[cls==cid]=0  ##待考察类
    label[cls!=cid]=1  ##其它类
    jobs=[]    
    res=[]
    objs=div_list( len_obj,n_parts )
    starts=div_list( len_ts-min_width,s_parts,s_step )
    
    start_t=time.time()
    #print starts
    #####对每个对象块列表进行循环，
    for o in objs:
        oparts=[]
        ####对象块中的对象进行循环，如果类0对象，则加入oparts         
        for oo in o:
            if label[oo]==0:
                oparts.append(oo)
        ####如果oparts非空，则循环每个可能开始位置start,并提交任务        
        if len(oparts) >= 1:
            for s in starts:
                jobs.append(job_server.submit(compute,(oparts,s,min_width,w_step,data,label),(),("numpy",)))
        #print oparts
    for job in jobs:
        res.append( job() )
    
    res=numpy.array(res)
    print res[numpy.argmax(res[:,4]),:]
    best_res=res[numpy.argmax(res[:,4]),:]
    model[count,0]=cid
    model[count,1:5]=best_res[1:]
    model[count,5:]=data[best_res[0],:]
    count+=1
    print job_server.print_stats()
    print time.time()-start_t

model=model[numpy.argsort(-1*model[:,4]),:]
numpy.savetxt("./data/model_"+fname,model,delimiter=",")  
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    