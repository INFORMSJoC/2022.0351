import sys
import arrow
import math
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import sklearn
import zss
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize, Bounds
import timeit
import scipy.integrate
from scipy.integrate import quad
from scipy.integrate import tplquad
import statistics
from sklearn.cluster import DBSCAN
from st_dbscan import ST_DBSCAN
from sklearn.cluster import AgglomerativeClustering
from minisom import MiniSom
from datetime import datetime
import time
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
from scipy.stats import norm
from numpy.linalg import inv, det
from scipy.spatial.distance import cdist
from scipy.integrate import nquad
import utils
from utils import STPPS, WeirdNode


try:
    from editdist import distance as strdist
except ImportError:
    def strdist(a, b):
        if a == b:
            return 0
        else:
            return 1

def weird_dist(A, B):
    return 10*strdist(A, B)


##### Simulation Setting
cte_config = dict()

cte_config['final-mu'] = False

cte_config['simulation'] = 1 # 1 - 5.1 2- 5.2
# cte_config['lam'] = [80,30]# = mu * X * Y * T
cte_config['lam'] = 20# = mu * X * Y * T
cte_config['alpha'] =10
cte_config['beta'] = 5
cte_config['sigx'] = 0.2
cte_config['sigy'] = 0.2

cte_config['gamma'] = 0.3

cte_config['round'] =  2# default = 1

##### Experiment Setup
cte_config['cte_'] = True  # cte
cte_config['cte_g'] = True  # cte_groundtruth
cte_config['em_cte'] = False    # em_cte
cte_config['em_r'] = False     # em_random
cte_config['em_g'] = False     # em_groundtruth
cte_config['mle'] = False  # mle_Nelder-Mead
cte_config['mle_nm'] = False  # mle_Nelder-Mead
cte_config['mle_cg'] = False # mle_CG
cte_config['mle_newton'] = False # mle_L-BFGS-B

cte_config['em_de'] = False  # em_declustering

cte_config['variance'] = False #simulation_variance
cte_config['variance_loop'] = 4 # samples for computing variance

##### CTE Parameter
cte_config['initial_guess'] = [10,5,0.2,0.2]
cte_config['initial_guess_mle'] = [20,10,5,0.15,0.15] # only for mle
cte_config['sampler_rate_1'] = 0.12#0.4 #0.3 # 0.20 -20 #0.12
cte_config['sampler_rate_2'] =0.69 # 0.64 # 0.6 - 20 #0.68

cte_config['cluster'] = 'd'  #  cte_config['cluster']  d - dbscan , st - stdbscan ,  h - agglomerative , s - som

cte_config['st-timedecay'] = 0.4  # ST_DBSCAN Para

cte_config['somsig'] = 0.1  # SOM Para
cte_config['somlr'] = 0.01  # SOM Para
cte_config['somx'] = 65 #75  # SOM Para
cte_config['somepoch'] = 10 #10  # SOM Para


##### EM
cte_config['sample_size'] = 2
cte_config['em_ini'] = 1
cte_config['threshold'] = 0.4


# ##### Dataset
cte_config['dataset'] = '' #''-simulation 'o'-onlien retail 'e'-earthquake 'c'-citibike '9'-911_crime
cte_config['select'] = 1000
cte_config['cte_kde'] = False
cte_config['em_kde'] = 0
cte_config['graph'] = False


def Sampler(data,Random,first):
    if cte_config['cluster'] == 'd' or cte_config['cluster'] == 'h':
        if data.shape[1] >3: ## delete data into raw dim (t,x,y)
            data = pp.del2data(data)
        dist = []
        data_len = len(data)
        if data_len == 2:
            onepair = (np.abs(data[0] - data[1])).sum()
            epsilon = np.abs(np.random.normal(onepair,0.05*onepair)) + 0.01
        elif data_len == 0:
            return 0.01
        else:
            for i in range(int(((data_len-2)/3)+3)):
                sam = np.random.randint(0,data_len,size = 2)
                if sam[0] - sam[1] == 0:
                    if sam[0] < data_len-1:
                        sam[0] = data_len-1
                    else:
                        sam[0] = 0
                # else:
                distpoint = abs((data[sam[0]] -data[sam[1]]).sum())
                if Random == 1:
                    if first == 1:
                        distpoint = (distpoint*(1 - (np.random.normal(0,0.1*distpoint)))) * cte_config['sampler_rate_1']
                    else:
                        distpoint = distpoint*(1 - np.abs(np.random.normal(0,0.1*distpoint))) * cte_config['sampler_rate_2']
                dist.append(distpoint)
            dist = np.abs(dist)
            epsilon = np.abs(np.mean(dist) + np.random.normal(0,0.1*np.mean(dist)))+ 0.01

    elif cte_config['cluster'] == 'st':
        epi = []
        if data.shape[1] >3: ## delete data into raw dim (t,x,y)
            data = pp.del2data(data)
        dist = []
        data_len = len(data)
        if data_len == 2:
            onepair = (np.abs(data[0] - data[1])).sum()
            epsilon = np.abs(np.random.normal(onepair,0.05*onepair)) + 0.01
        elif data_len == 0:
            # print('LENNNNN!')
            return 0.01
        else:
            for i in range(int(((data_len-2)/3)+3)):
                sam = np.random.randint(0,data_len,size = 2)
                if sam[0] - sam[1] == 0:
                    if sam[0] < data_len-1:
                        sam[0] = data_len-1
                    else:
                        sam[0] = 0
                # else:
                distpoint = abs((data[sam[0]] -data[sam[1]]).sum())
                if Random == 1:
                    if first == 1:
                        distpoint = (distpoint*(1 - (np.random.normal(0,0.1*distpoint)))) * cte_config['sampler_rate_1']
                    else:
                        distpoint = distpoint*(1 - np.abs(np.random.normal(0,0.1*distpoint))) * cte_config['sampler_rate_2']
                dist.append(distpoint)
            dist = np.abs(dist)
            epsilon = np.abs(np.mean(dist) + np.random.normal(0,0.1*np.mean(dist)))+ 0.01

        epi.append(epsilon * cte_config['st-timedecay'])
        epi.append(epsilon)
        return epi


    else:   ## SOM Sampler
        if first == 1:
            epsilon = cte_config['sampler_rate_1']
        else:
            epsilon = cte_config['sampler_rate_2']
    return epsilon + 0.01


def emtree(em_P_mat,sample_time):
    ## Sampling
    shape = em_P_mat.shape[0]
    sample = []
    for i in range(sample_time):
        sample_P_mat = np.zeros((shape,shape))
        for j in range(shape):
            row_sample = np.random.choice(shape, 1, replace=False, p = list(em_P_mat[j,:]))
            sample_P_mat[j,row_sample[0]] = 1
        sample.append(sample_P_mat)
    # for jj in sample:
    #     print(jj.shape)
    tree = []
    ## Mat2Tree for zss
    for i in range(len(sample)):
        alphabet = []
        mat = sample[i]
        first = np.array([0])
        rest = []
        for j in range(shape):
            if mat[j,j] == 1:
                first = np.r_[first,np.array([j])]
            else:
                for k in range(j):
                    if mat[j,k] == 1:
                        rest.append(np.array([[j,k]]))
        first = np.delete(first,0)
        alphabet.append(first)
        ## first generation complete then process next generations
        if rest:
            index = []
            index1 = np.array([0])
            nextg = np.array([0])
            rest_nextg = np.array([[0,0]])
            # print(rest[1][0,1])
            for j in rest:
                if j[0,1] in first:
                    nextg = np.r_[nextg,np.array([j[0,0]])]
                    index1 = np.r_[index1,np.array([j[0,1]])]
                else:
                    rest_nextg = np.concatenate((rest_nextg,j),1)
            nextg = np.delete(nextg,0)
            index1 = np.delete(index1,0)
            index.append(index1)
            alphabet.append(nextg)
            resent_nextg = rest_nextg
            ## stop until finish the rest
            while len(rest_nextg) > 1:
                rest_nextg = np.array([[0,0]])
                local_first = alphabet[-1]
                local_current = np.array([0])
                local_index1 = np.array([0])
                for item in range(1,len(resent_nextg)): ## haven't delete first elemnt for resent_nextg
                    # print(local_first)
                    # print(resent_nextg)
                    if resent_nextg[item][0,1] in local_first:
                        local_current = np.r_[local_current,np.array([resent_nextg[item][0,0]])]
                        local_index1 = np.r_[local_index1,np.array([resent_nextg[item][0,1]])]
                    else:
                        rest_nextg = np.concatenate((rest_nextg,np.array(resent_nextg[item])),1)
                local_current = np.delete(local_current ,0)
                local_index1 = np.delete(local_index1 ,0)
                alphabet.append(local_current)
                index.append(local_index1)
                resent_nextg = rest_nextg
            item_tree = BuildTree(alphabet)
            # print(len(alphabet))
            # print(len(index))
            item_tree = Array2Tree_Infer(alphabet,index,item_tree)
            tree.append(item_tree)
        else:
            item_tree = BuildTree(alphabet)
            item_tree = Array2Tree_Infer_only(alphabet,item_tree)
            tree.append(item_tree)
    return tree








def index2real(index):
    long = len(index)
    for i in range(1,long):
        for j in range(len(index[i])):
            index[i][j] = index[i][j] + Alphabet[i-1][-1]


def SortArray(Datalist,data):
    Sort = []
    lenlist = len(Datalist)
    # Start = 1
    # lenIte = 0
    for i in range(lenlist):
        subsort = np.zeros((Datalist[i].shape[0]))
        for j in range(Datalist[i].shape[0]):
            for k in range(len(data)):
                if Datalist[i][j][0] == data[k][0]:
                    if Datalist[i][j][1] == data[k][1]:
                        subsort[j] = k
        Sort.append(subsort)
        # lenIte = Datalist[i].shape[0] + Start
        # itemshape = np.r_[Start:lenIte:1]
        # Start = Start + itemshape.shape[0]
        # Sort.append(itemshape)
    return Sort



def BuildTree(Index):
    Tree = []
    for i in Index:
        for j in i:
            Tree.append(WeirdNode(int(j)))
    return Tree


def Array2Tree(Data,Index1,Tree):#Alphabet,Index,Tree
    lenInx = len(Index1)-1
    #lenData = len(Data)-1
    for i in range(lenInx,-1,-1):
        InxItem = Index1[i]
        DataItem = Data[i+1] # pair data and index
        DataItemParent = Data[i] #pair data parent
        for j in range(len(InxItem)):
            Parent = int(DataItemParent[int(InxItem[j]-1)]-1) # inner -1 ---the index for Index pointer outer -1 index for parent
            Child = int(DataItem[j]-1)
            for h in range(len(Tree)):
                for l in range(len(Tree)):
                    if Tree[h].my_label == Parent and Tree[l].my_label == Child:
                        Tree[h].addkid(Tree[l])
    A = WeirdNode(-1)
    Background = Data[0]
    for k in Background:
        for h in range(len(Tree)):
            if Tree[h].my_label == k:
                A.addkid(Tree[h])
    return A

def Array2Tree_Infer(Data,Index1,Tree1): # All_Infer_int,Index_Infer,Tree_Infer
    lenInx = len(Index1)-1
    #lenData = len(Data)-1
    for i in range(lenInx,-1,-1):
        InxItem = Index1[i] # current generation of their index
        DataItem = Data[i+1] # current index's corresponding data
        #DataItemParent = Data[i]
        for j in range(len(InxItem)):
            Parent = int(InxItem[j]-1) # -1 because its index
            Child = int(DataItem[j]-1) # -1 because its index
            for h in range(len(Tree)):
                for l in range(len(Tree)):
                    if Tree[h].my_label == Parent and Tree[l].my_label == Child:
                        Tree[h].addkid(Tree[l])
                        # Tree1[Parent].addkid(Tree1[Child])
    A = WeirdNode(-1)
    Background = Data[0]
    for k in Background:
        for h in range(len(Tree)):
            if Tree[h].my_label == k:
                A.addkid(Tree[h])
    return A

def Array2Tree_Infer_only(Data,Tree1): # All_Infer_int,Index_Infer,Tree_Infer
    # lenInx = len(Index1)-1
    # #lenData = len(Data)-1
    # for i in range(lenInx,-1,-1):
    #     InxItem = Index1[i] # current generation of their index
    #     DataItem = Data[i+1] # current index's corresponding data
    #     #DataItemParent = Data[i]
    #     for j in range(len(InxItem)):
    #         Parent = int(InxItem[j]-1) # -1 because its index
    #         Child = int(DataItem[j]-1) # -1 because its index
    #         for h in range(len(Tree)):
    #             for l in range(len(Tree)):
    #                 if Tree[h].my_label == Parent and Tree[l].my_label == Child:
    #                     Tree[h].addkid(Tree[l])
    #                     # Tree1[Parent].addkid(Tree1[Child])
    A = WeirdNode(-1)
    Background = Data[0]
    for k in Background:
        for h in range(len(Tree)):
            if Tree[h].my_label == k:
                A.addkid(Tree[h])
    return A


def SearchLabel(Data,Sheet):  #After Clustering
    Label4Data = np.zeros((1))
    for i in range(len(Data)):
        for j in Sheet:
            if Data[i][1] == j[1]:
                if Data[i][2] == j[2]:
                    if Data[i][0] == j[0]:
                        Label4Data = np.concatenate((Label4Data,np.array([j[-1]])),axis = 0)
    Label4Data = np.delete(Label4Data,[0],axis = 0)
    #Data = np.concatenate((Data,Label4Data), axis = 1)
    return Label4Data


def discrete_sum(data,index_infer,all_infer):
    X_all = np.zeros((1))
    Y_all = np.zeros((1))
    T_all = np.zeros((1))
    lenInx = len(index_infer)-1
    #lenData = len(Data)-1
    for i in range(lenInx,-1,-1):
        InxItem = index_infer[i]
        DataItem = all_infer[i+1]
        #DataItemParent = Data[i]
        for j in range(len(InxItem)):
            Parent = int(InxItem[j]-1) ### index = num -1
            Child = int(DataItem[j]-1)
            X = data[Child][1]-data[Parent][1]
            Y = data[Child][2]-data[Parent][2]
            T = data[Child][0]-data[Parent][0]
            X_all = np.concatenate((X_all,np.array([X])),axis = 0)
            Y_all = np.concatenate((Y_all,np.array([Y])),axis = 0)
            T_all = np.concatenate((T_all,np.array([T])),axis = 0)
    X_all = np.delete(X_all,[0],axis = 0)
    Y_all = np.delete(Y_all,[0],axis = 0)
    T_all = np.delete(T_all,[0],axis = 0)
    return X_all, Y_all, T_all



def clustering_pipeline(raw_data_clustering,Real_data):
    if len(raw_data_clustering.shape) > 2:
        raw_data_clustering = raw_data_clustering.reshape(raw_data_clustering.shape[-2],raw_data_clustering.shape[-1])
    raw_data_clustering = pp.del2data(raw_data_clustering)
    label_raw_data_clustering = pp.dbscan(raw_data_clustering, Sampler(raw_data_clustering,1,0))
    label_raw_data_clustering = label_raw_data_clustering.reshape((len(label_raw_data_clustering),1))
    raw_data_clustering = np.concatenate((raw_data_clustering,label_raw_data_clustering),axis = 1)  ##add clustering label
    Index2 = SearchLabel(raw_data_clustering,Real_data)  ##Search raw id
    Index2 = Index2.reshape((len(Index2),1))
    raw_data_clustering = np.concatenate((raw_data_clustering,Index2),axis = 1)
    return raw_data_clustering




def em_decluster(P):
    sample_P_mat = np.zeros((P.shape[0],P.shape[0]))
    for i in range(P.shape[0]):
        row_sample = np.random.choice(i+1, 1, replace=False, p = P[i,:i+1])
        sample_P_mat[i, row_sample[0]] = 1
    return sample_P_mat


if cte_config['cte_']:
    cte_mu = []
    cte_alpha = []
    cte_beta = []
    cte_sig1 = []
    cte_sig2 = []
    cte = []

if cte_config['cte_g']:
    cteg_mu = []
    cteg_alpha = []
    cteg_beta = []
    cteg_sig1 = []
    cteg_sig2 = []
    cteg = []

if cte_config['em_cte']:
    em_mu = []
    em_alpha = []
    em_beta = []
    em_sig1 = []
    em_sig2 = []
    em_dist = []
    em = []

if cte_config['em_r']:
    emr_mu = []
    emr_alpha = []
    emr_beta = []
    emr_sig1 = []
    emr_sig2 = []
    emr_dist = []
    emr = []

if cte_config['em_g']:
    emg_mu = []
    emg_alpha = []
    emg_beta = []
    emg_sig1 = []
    emg_sig2 = []
    emg_dist = []
    emg = []

if cte_config['em_de']:
    emde_mu = []
    emde_alpha = []
    emde_beta = []
    emde_sig1 = []
    emde_sig2 = []
    emde_dist = []
    emde = []

if cte_config['mle']: #no_tree_dist
    mle_mu = []
    mle_alpha = []
    mle_beta = []
    mle_sig1 = []
    mle_sig2 = []
    # emde_dist = []
    mle = []


start = timeit.default_timer()

dist_list = []

llh = []

finalmu = []

if cte_config['em_kde'] == 0:
    print('Main Part')
    logseq = []
    for i in range(cte_config['round']):
        if cte_config['simulation'] == 1:
            pp = STPPS(cte_config['lam'],cte_config['alpha'],cte_config['beta'],cte_config['sigx'],cte_config['sigy'], cte_config)
            Ini = pp.InitialGenerate(T=[0,10], S=[[-5, 5], [-5, 5]])
            Ini1 = Ini
            All = [Ini]
            Index = []
            decay = 1
            while True:
                Ini1, PC = pp.OffspringPropagation(Ini,decay)
                if Ini1.shape == (0,3):
                    break
                else:
                    All.append(Ini1)
                    Ini = Ini1
                    Index.append(PC)
            
    
        if cte_config['simulation'] == 2:
            pp = STPPS2(cte_config['lam'],cte_config['alpha'],cte_config['beta'],cte_config['gamma'],cte_config['sigx'],cte_config['sigy'], cte_config)
            Ini = pp.InitialGenerate(T=[0, 10], S=[[-5, 5], [-5, 5]])
            Ini1 = Ini
            All = [Ini]
            Index = []
            decay = 1
            while True:
                Ini1, PC = pp.OffspringPropagation(Ini,decay)
                if Ini1.shape == (0,3):
                    break
                else:
                    All.append(Ini1)
                    Ini = Ini1
                    Index.append(PC)
    
        # pp = STPPS(50,10,5,0.6,0.6)
        # Ini = pp.InitialGenerate(T=[0, 10], S=[[-5, 5], [-5, 5]])
        # Ini1 = Ini
        # All = [Ini]
        # Index = []
        # decay = 1
        # while True:
        #     Ini1, PC = pp.OffspringPropagation(Ini,decay)
        #     if Ini1.shape == (0,3):
        #         break
        #     else:
        #         All.append(Ini1)
        #         Ini = Ini1
        #         Index.append(PC)
        #     #decay += 1
    
    
        #### First Round Clustering
        if cte_config['dataset']:
            data = dataset()
        else:
            data = pp.list2array(All)
        label = pp.dbscan(data,Sampler(data,1,1))
    
    
    
        label = label.reshape((len(label),1))
        idid = np.array([range(1,len(label)+1)]).reshape((len(label),1))
        data = np.concatenate((data,label),axis = 1)
        data = np.concatenate((data,idid),axis = 1) ## add clustering label and id
    
        # print(Index)
    

    
    
        epoch = 1
        x_est = []
        y_est = []
        alpha_est = []
        beta_est = []
        x_est_1 = []
        y_est_1 = []
        alpha_est_1 = []
        beta_est_1 = []
        # dist_list = []
        lam_list = []
        lam_list_1 = []
    
        '''
        # for epoch in range(1):
            #### Simulation
        pp = STPPS(50,10,5,0.1,0.1)
        Ini = pp.InitialGenerate(T=[0, 10], S=[[-5, 5], [-5, 5]])
        Ini1 = Ini
        All = [Ini]
        Index = []
        decay = 1
        while True:
            Ini1, PC = pp.OffspringPropagation(Ini,decay)
            if Ini1.shape == (0,3):
                break
            else:
                All.append(Ini1)
                Ini = Ini1
                Index.append(PC)
    
    
        # pp = STPPS(50,10,5,0.6,0.6)
        # Ini = pp.InitialGenerate(T=[0, 10], S=[[-5, 5], [-5, 5]])
        # Ini1 = Ini
        # All = [Ini]
        # Index = []
        # decay = 1
        # while True:
        #     Ini1, PC = pp.OffspringPropagation(Ini,decay)
        #     if Ini1.shape == (0,3):
        #         break
        #     else:
        #         All.append(Ini1)
        #         Ini = Ini1
        #         Index.append(PC)
        #     #decay += 1
    
    
        #### First Round Clustering
        data = pp.list2array(All)
        label = pp.dbscan(data,Sampler(data,1,1))
        label = label.reshape((len(label),1))
        idid = np.array([range(1,len(label)+1)]).reshape((len(label),1))
        data = np.concatenate((data,label),axis = 1)
        data = np.concatenate((data,idid),axis = 1) ## add clustering label and id
    
        '''
    
        # #### DataFrame
        # df = pd.DataFrame(data)
        # df['id'] = range(1,len(df)+1)
    
    
        #### Infer Frame Data
        All_Infer = []
        Index_Infer = []
        cache = []
        delcenter,groupind,i1,i2,ind,bol = pp.del_center1(data)
        All_Infer.append(i1)
        Index_Infer.append(ind)
        All_Infer.append(i2)
    
    
    
    
    
        ### Second Round
    
        starting_generation_index = 1
    
        stage_all1 = np.zeros((1))
        stage_all2 = np.zeros((1))
        stage_index1 = np.zeros((1))
        stage_index2 = np.zeros((1))
        del2 = []
        gIndex = []
        breakcriteria = []
        for i in range(len(delcenter)):
            delcentersub = pp.del2data(delcenter[i])  #to raw data for clustering
            label2 = pp.dbscan(delcentersub,Sampler(delcentersub,1,0))  #clustering
            label2 = label2.reshape((len(label2),1))
            delcentersub = np.concatenate((delcentersub,label2),axis = 1)  ##add clustering label
            Index2 = SearchLabel(delcentersub,data)  ##Search raw id
            Index2 = Index2.reshape((len(Index2),1))
            #print(Index2.shape)
            #print(delcentersub.shape)
            delcentersub = np.concatenate((delcentersub,Index2),axis = 1)  ##add raw id
            delcenter2,groupind2,i12,i22,ind1,ind2,bol = pp.del_center(delcentersub,groupind[i]) #delcenter
            stage_all1 = np.concatenate((stage_all1,i12), axis =0)
            stage_all2 = np.concatenate((stage_all2,i22), axis =0)
            stage_index1 = np.concatenate((stage_index1,ind1), axis =0)
            stage_index2 = np.concatenate((stage_index2,ind2), axis =0)
            ## delcenter2 / groupindex storage
            delcenter2 = np.array(delcenter2)
            groupind2 = np.array(groupind2)
            del2.append(delcenter2)
            gIndex.append(groupind2)
            breakcriteria.append(bol)
        stage_all1 = np.delete(stage_all1,[0],axis = 0)
        stage_all2 = np.delete(stage_all2,[0],axis = 0)
        stage_index1 = np.delete(stage_index1,[0],axis = 0)
        stage_index2 = np.delete(stage_index2,[0],axis = 0)
        All_Infer[-1] = np.concatenate((All_Infer[-1],stage_all1), axis =0)
        All_Infer.append(stage_all2)
        Index_Infer[-1] = np.concatenate((Index_Infer[-1],stage_index1), axis =0)
        Index_Infer.append(stage_index2)
    
    
        ### after second round
        #  if All_Infer[-1].shape[0] == 0:
        #     All_Infer.pop()
        #    Index_Infer.pop()
    
        #### third round
        breakcrit = [False]
        #eps = [1.3,1.1,0.9,0.8,0.6]
        while all(breakcrit) == False:
            genera =0
            stage_all1 = np.zeros((1))
            stage_all2 = np.zeros((1))
            stage_index1 = np.zeros((1))
            stage_index2 = np.zeros((1))
            del3 = []
            gIndex3 = []
            breakcrit = []
            for i in range(len(del2)):
    
                if len(del2[i].shape) > 1:  ### single or multi array
                    if del2[i].shape[0] == 1:
                        subdel2 = clustering_pipeline(del2[i],data)
                        sub3,group3,i13,i23,ind13,ind23,bol = pp.del_center(subdel2,int(gIndex[i]))
                        stage_all1 = np.concatenate((stage_all1,i13), axis =0)
                        stage_all2 = np.concatenate((stage_all2,i23), axis =0)
                        stage_index1 = np.concatenate((stage_index1,ind13), axis =0)
                        stage_index2 = np.concatenate((stage_index2,ind23), axis =0)
                        sub3 = np.array(sub3)
                        group3 = np.array(group3)
                        del3.append(sub3)
                        gIndex3.append(group3)
                        breakcrit.append(bol)
                    else:
                        for j in range(del2[i].shape[0]):
                            subdel2 = clustering_pipeline(del2[i][j],data)
                            sub3,group3,i13,i23,ind13,ind23,bol = pp.del_center(subdel2,int(gIndex[i][j]))
                            stage_all1 = np.concatenate((stage_all1,i13), axis =0)
                            stage_all2 = np.concatenate((stage_all2,i23), axis =0)
                            stage_index1 = np.concatenate((stage_index1,ind13), axis =0)
                            stage_index2 = np.concatenate((stage_index2,ind23), axis =0)
                            sub3 = np.array(sub3)
                            group3 = np.array(group3)
                            del3.append(sub3)
                            gIndex3.append(group3)
                            breakcrit.append(bol)
                else:
                    del2_sub = del2[i].tolist()
                    for k in range(len(del2_sub)):
                        subdel2 = clustering_pipeline(del2_sub[k],data)
                        sub3,group3,i13,i23,ind13,ind23,bol = pp.del_center(subdel2,int(gIndex[i][k]))
                        stage_all1 = np.concatenate((stage_all1,i13), axis =0)
                        stage_all2 = np.concatenate((stage_all2,i23), axis =0)
                        stage_index1 = np.concatenate((stage_index1,ind13), axis =0)
                        stage_index2 = np.concatenate((stage_index2,ind23), axis =0)
                        sub3 = np.array(sub3)
                        group3 = np.array(group3)
                        del3.append(sub3)
                        gIndex3.append(group3)
                        breakcrit.append(bol)
    
    
            stage_all1 = np.delete(stage_all1,[0],axis = 0)
            stage_all2 = np.delete(stage_all2,[0],axis = 0)
            stage_index1 = np.delete(stage_index1,[0],axis = 0)
            stage_index2 = np.delete(stage_index2,[0],axis = 0)
            All_Infer[-1] = np.concatenate((All_Infer[-1],stage_all1), axis =0)
            All_Infer.append(stage_all2)
            Index_Infer[-1] = np.concatenate((Index_Infer[-1],stage_index1), axis =0)
            Index_Infer.append(stage_index2)
            del2 = del3
            gIndex = gIndex3
    
        if All_Infer[-1].shape[0] == 0:
          All_Infer.pop()
          Index_Infer.pop()
    
        ### list to array then to list
    
    
    
    
        ### log likelihood scipy optimise
    
    
    
    
    
    
        Alphabet = SortArray(All,data)
        Tree = BuildTree(Alphabet)
        J= Array2Tree(Alphabet,Index,Tree)
    
        misA = Alphabet[0]
        misB = All_Infer[0]
        ucount = np.unique(np.concatenate((np.setdiff1d(misA, misB), np.setdiff1d(misB, misA)))).size
        logseq.append(ucount)
        
        #Alphabet_Infer = SortArray(All_Infer)
        ## from float to int
        All_Infer_int = []
        for num in All_Infer:
          num = num.astype(int)
          All_Infer_int.append(num)
    
        Tree_Infer = BuildTree(All_Infer_int)
        K = Array2Tree_Infer(All_Infer_int,Index_Infer,Tree_Infer)
    
        dist = zss.simple_distance(J, K, WeirdNode.get_children, WeirdNode.get_label, weird_dist)
    
        dist_list.append(dist)
    
    
        def Index2P(data,index,allinfer):
            shape = data.shape[0]
            background = allinfer[0]
            P_Infer = np.zeros((shape,shape))
            for i in range(len(allinfer)):
                if i == 0:
                    for j in range(background.shape[0]):
                        P_Infer[int(background[j]-1)][int(background[j]-1)] = 1
                else:
                    for j in range(allinfer[i].shape[0]):
                        P_Infer[int(allinfer[i][j]-1)][int(index[i-1][j]-1)] = 1
            return P_Infer
    
        P_mat = Index2P(data,Index_Infer,All_Infer)
    
    
        def _random_init_P(n_point):
            P = np.random.uniform(low=0., high=1., size=(n_point, n_point))
            for i in range(n_point):
                if i == 0:
                    P[i, :i+1] = 1/(i+1)
                    # P[i,  i] = 1 - (i-1)*(1/(cte_config['em_ini']*(i+1)))
                    P[i, i+1:] = 0
                else:
                    P[i, :i] = 1/(i**2)
                    P[i, i] = 1 - (i-1)/(i**2)
                    P[i, i+1:] = 0
            return P
    
        def update_P(mat,param,data):
            for i in range(1,mat.shape[0]):
                bottom = lam(param,i)
                for j in range(i+1):
                    if j < i:
                        up = g(param[0],param[1],param[2],param[3],data,i,j)
                        # print(up)
                        mat[i,j] = up/bottom
                    elif j == i:
                        mat[i,j] = 1 - mat[i,:j].sum()
                    else:
                        mat[i,j] = 0
            return mat
    
    
        # def realP(data,alll,indexx):
        #     shape = data.shape[0]
        #     print(shape)
        #     raww = np.zeros((shape,shape))
        #     for i in range(len(alll)):
        #         for j in range(len(alll[i])):
        #             if i == 0:
        #                 for k in range(shape):
        #                     if alll[i][j][0] == data[k][0]:
        #                         raww[k][k] = 1
        #             else:
        #                 for k in range(shape):
        #                     ind = int(indexx[i-1][j])
        #                     if alll[i-1][ind-1][0] == data[k][0]:
        #                         for h in range(shape):
        #                             if alll[i][j][0] == data[h][0]:
        #                                 raww[h][k] = 1
        #     return raww
    
        # def count(data,itemi):
        #     leng = data.shape[0]
        #     num = 0
        #     for i in range(leng):
        #         if data[itemi][0] > data[i][0]:
        #             num = num + 1
        #     return num
    
    
        def realP(data,alll,indexx):
            shape = data.shape[0]
            #print(shape)
            raww = np.zeros((shape,shape))
            for i in range(len(alll)):
                for j in range(len(alll[i])):
                    if i == 0:
                        for k in range(shape):
                            if alll[i][j][0] == data[k][0]:
                                # count = int(count(data,k))
                                raww[k][k] = 1
                    else:
                        for k in range(shape):
                            ind = int(indexx[i-1][j])
                            if alll[i-1][ind-1][0] == data[k][0]:
                                # countk = int(count(data,k))
                                # check parent id
                                for h in range(shape):
                                    if alll[i][j][0] == data[h][0]: # check child id
                                        # counth = int(count(data,h))
                                        raww[h][k] = 1
            return raww
    
    
        def g(alpha,beta,sigma1,sigma2,data,itemi,itemj):
            value = alpha * np.exp( - beta * (data[itemi][0] - data[itemj][0]) - 0.5 * (((data[itemi][1]-data[itemj][1])**2/(sigma1**2)) + ((data[itemi][2]-data[itemj][2])**2/(sigma2**2))) )
            if value < 10**(-9):
                value = 0.0000001
            return value
    
        def mu(Ti,Xi,Yi):
            # return np.trace(P_mat)/1000
            return cte_config['lam']/1000
        # def lam(param,itemi):
        #     lam = mu(10,10,10) ## ASSU CORT
        #     for i in range(itemi):
        #         for j in range(i+1):
        #             lam = lam + g(param[0],param[1],param[2],param[3],data,j,i)
        #     return lam
    
        def lam(param,itemi):
            lam = mu(10,10,10) ## ASSU CORT
            for j in range(P_mat.shape[0]):
                if data[itemi][0] > data[j][0]:
                    lam = lam + g(param[0],param[1],param[2],param[3],data,itemi,j)
            return lam
    
    
    
        def loglikihood(params):
            alpha,beta,sigma1,sigma2 = params[0],params[1],params[2],params[3]
            ### log alpha
            term1 = 0
            for i in range(P_mat.shape[0]):
                term1 = term1 + P_mat[i][i] * np.log(mu(data[i][0],data[i][1],data[i][2]))
            ### log g
            term2 = 0
            for i in range(P_mat.shape[0]):
                for j in range(i):
                    term2 = term2 + P_mat[i][j] * np.log(g(alpha,beta,sigma1,sigma2,data,i,j))
            ### integral laam
            term3 = np.trace(P_mat)
            for i in range(P_mat.shape[0]):
                term31 =  quad(lambda t: alpha* np.exp(- beta * (t - data[i][0])),data[i][0], 10  )
                term32 =  quad(lambda x: np.exp(- 0.5 * (((x - data[i][1])**2/(sigma1**2)))), -5,5 )
                term33 =  quad(lambda y: np.exp(- 0.5 * (((y - data[i][2])**2/(sigma2**2)))), -5,5 )
                # term3 = term3 + tplquad(lambda x,y,z: alpha * math.exp( - beta * (x - data[i][0]) - 0.5 * (((y - data[i][1])**2/(sigma1**2)) + ((z - data[i][2])**2/(sigma2**2))) ), data[i][0],10,-5,5,-5,5)
                term3 = term3 + term31[0] * term32[0] * term33[0]
            final =  -term1 - term2 + term3
            return final
    
    
    
        def safe_exp(x):
            MAX_VAL = np.log(np.finfo('float').max)
            return np.exp(x if x < MAX_VAL else MAX_VAL)
    
        def g_new(alpha,beta,sigma1,sigma2,dataa,itemi):
            if itemi == 0:
              return 0
            value = 0
            for ago in range(itemi):
                subvalue = alpha * safe_exp( - beta * (dataa[itemi][0] - dataa[ago][0]) - 0.5 * (((dataa[itemi][1]-dataa[ago][1])**2/(sigma1**2)) + ((dataa[itemi][2]-dataa[ago][2])**2/(sigma2**2))) )
                if subvalue < 10**(-6):
                    subvalue = 0.00001
                value += subvalue
            return value
    
    
        def loglikihood_new(params):
            mu,alpha,beta,sigma1,sigma2 = params[0],params[1],params[2],params[3], params[4]
            term2 = 0
            for i in range(data.shape[0]):
                term2 = term2 + np.log(mu/1000 + g_new(alpha,beta,sigma1,sigma2,data,i))
    
            term3 = mu
            for i in range(data.shape[0]):
                term31 =  quad(lambda t: alpha* safe_exp(- beta * (t - data[i][0])),data[i][0], 10  )
                term32 =  quad(lambda x: safe_exp(- 0.5 * (((x - data[i][1])**2/(sigma1**2)))), -5,5 )
                term33 =  quad(lambda y: safe_exp(- 0.5 * (((y - data[i][2])**2/(sigma2**2)))), -5,5 )
    
                term3 = term3 + term31[0] * term32[0] * term33[0]
    
            final =  - term2 + term3
            return final
    
    
    
    
        if cte_config['final-mu']:
            P_mat = Index2P(data,Index_Infer,All_Infer)
            finalmu.append(np.trace(P_mat))
    
    
    
        bounds = Bounds([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf])
        bounds_mle = Bounds([0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf])
    
        
    
    
        if cte_config['cte_']:
            P_mat = Index2P(data,Index_Infer,All_Infer)
            xa = cte_config['initial_guess']
            result2 = minimize(fun = loglikihood, x0 = xa, method = 'Nelder-Mead', bounds=bounds)
            print(result2.fun)
            # print(result2)
            cte_mu.append(np.trace(P_mat))
            cte_alpha.append(result2.x[0])
            cte_beta.append(result2.x[1])
            cte_sig1.append(result2.x[2])
            cte_sig2.append(result2.x[3])
            # print('log-likelihood',result2.fun)
            llh.append(result2.fun)
    
        if cte_config['cte_g']:
            P_mat = realP(data,All,Index)
            # P_mat = _random_init_P(data.shape[0])
            xa = cte_config['initial_guess']
            result1 = minimize(fun = loglikihood, x0 = xa,method = 'Nelder-Mead', bounds=bounds)
            # print(result1)
            cteg_mu.append(np.trace(P_mat))
            cteg_alpha.append(result1.x[0])
            cteg_beta.append(result1.x[1])
            cteg_sig1.append(result1.x[2])
            cteg_sig2.append(result1.x[3])
            # print('log-likelihood',result1.fun)
            llh.append(result1.fun)
    
        if cte_config['em_cte']:
            # P_mat = _random_init_P(data.shape[0])
            P_mat = Index2P(data,Index_Infer,All_Infer)
            # P_mat = realP(data,All,Index)
            xa = cte_config['initial_guess']
            result = minimize(fun = loglikihood, x0 = xa,method = 'Nelder-Mead', bounds=bounds)
            loss = 1
            while loss> cte_config['threshold']:
                result1 = minimize(fun = loglikihood, x0 = result.x,method = 'Nelder-Mead', bounds=bounds)
                P_mat = update_P(P_mat,result1.x,data)
                loss = abs((result.x - result1.x).sum())
                result = result1
            # print(result)
            # print('111')
            EM = emtree(P_mat, cte_config['sample_size'])
            for item in EM:
                dist = zss.simple_distance(J, item, WeirdNode.get_children, WeirdNode.get_label, weird_dist)
                em_dist.append(dist)
            em_mu.append(np.trace(P_mat))
            em_alpha.append(result.x[0])
            em_beta.append(result.x[1])
            em_sig1.append(result.x[2])
            em_sig2.append(result.x[3])
            print('log-likelihood',result.fun)
            llh.append(result.fun)
    
        if cte_config['em_r']:
            P_mat = _random_init_P(data.shape[0])
            # P_mat = Index2P(data,Index_Infer,All_Infer)
            # P_mat = realP(data,All,Index)
            xa = cte_config['initial_guess']
            result = minimize(fun = loglikihood, x0 = xa,method = 'Nelder-Mead', bounds=bounds)
            loss = 1
            while loss> cte_config['threshold']:
                result1 = minimize(fun = loglikihood, x0 = result.x,method = 'Nelder-Mead', bounds=bounds)
                P_mat = update_P(P_mat,result1.x,data)
                loss = abs((result.x - result1.x).sum())
                result = result1
            # print(result)
            # print('111')
            EM = emtree(P_mat, cte_config['sample_size'])
            for item in EM:
                dist = zss.simple_distance(J, item, WeirdNode.get_children, WeirdNode.get_label, weird_dist)
                emr_dist.append(dist)
            emr_mu.append(np.trace(P_mat))
            emr_alpha.append(result.x[0])
            emr_beta.append(result.x[1])
            emr_sig1.append(result.x[2])
            emr_sig2.append(result.x[3])
            print('log-likelihood',result.fun)
            llh.append(result.fun)
    
        if cte_config['em_g']:
            P_mat = realP(data,All,Index)
            xa = cte_config['initial_guess']
            result = minimize(fun = loglikihood, x0 = xa,method = 'Nelder-Mead', bounds=bounds)
            loss = 1
            while loss> cte_config['threshold']:
                result1 = minimize(fun = loglikihood, x0 = result.x,method = 'Nelder-Mead', bounds=bounds)
                P_mat = update_P(P_mat,result1.x,data)
                loss = abs((result.x - result1.x).sum())
                # print(loss)
                result = result1
            # print(result)
            # print('111')
            EM = emtree(P_mat, cte_config['sample_size'])
            for item in EM:
                dist = zss.simple_distance(J, item, WeirdNode.get_children, WeirdNode.get_label, weird_dist)
                emg_dist.append(dist)
            emg_mu.append(np.trace(P_mat))
            emg_alpha.append(result.x[0])
            emg_beta.append(result.x[1])
            emg_sig1.append(result.x[2])
            emg_sig2.append(result.x[3])
            print('log-likelihood',result.fun)
            llh.append(result.fun)
    
        if cte_config['em_de']:
            P_mat = _random_init_P(data.shape[0])
            xa = cte_config['initial_guess']
            result = minimize(fun = loglikihood, x0 = xa,method = 'Nelder-Mead', bounds=bounds)
            loss = 1
            while loss> cte_config['threshold']:
                result1 = minimize(fun = loglikihood, x0 = result.x,method = 'Nelder-Mead', bounds=bounds)
                P_mat = update_P(P_mat,result1.x,data)
                P_mat = em_decluster(P_mat)
                loss = abs((result.x - result1.x).sum())
                # print(loss)
                result = result1
            # print(result)
            # print('111')
            EM = emtree(P_mat, cte_config['sample_size'])
            for item in EM:
                dist = zss.simple_distance(J, item, WeirdNode.get_children, WeirdNode.get_label, weird_dist)
                emde_dist.append(dist)
            emde_mu.append(np.trace(P_mat))
            emde_alpha.append(result.x[0])
            emde_beta.append(result.x[1])
            emde_sig1.append(result.x[2])
            emde_sig2.append(result.x[3])
            print('log-likelihood',result.fun)
            llh.append(result.fun)
    
        if cte_config['mle']:
            xa = cte_config['initial_guess_mle']
            if cte_config['mle_nm']:
                print('mle_nm')
                result = minimize(fun = loglikihood_new, x0 = xa, method = 'Nelder-Mead', bounds=bounds_mle)
            if cte_config['mle_cg']:
                print('mle_cg')
                result = minimize(fun = loglikihood_new, x0 = xa, method = 'CG', bounds=bounds_mle)
            if cte_config['mle_newton']:
                print('mle_newton')
                result = minimize(fun = loglikihood_new, x0 = xa, method = 'L-BFGS-B', bounds=bounds_mle)
    
            mle_mu.append(result.x[0])
            mle_alpha.append(result.x[1])
            mle_beta.append(result.x[2])
            mle_sig1.append(result.x[3])
            mle_sig2.append(result.x[4])
            print('log-likelihood',result.fun)
            llh.append(result.fun)
    
    if cte_config['cte_']:
        cte.append(np.mean(cte_mu))
        cte.append(np.mean(cte_alpha))
        cte.append(np.mean(cte_beta))
        cte.append(np.mean(cte_sig1))
        cte.append(np.mean(cte_sig2))
        cte.append(np.mean(dist_list))
    
    if cte_config['cte_g']:
        cteg.append(np.mean(cteg_mu))
        cteg.append(np.mean(cteg_alpha))
        cteg.append(np.mean(cteg_beta))
        cteg.append(np.mean(cteg_sig1))
        cteg.append(np.mean(cteg_sig2))
    
    if cte_config['em_cte']:
        em.append(np.mean(em_mu))
        em.append(np.mean(em_alpha))
        em.append(np.mean(em_beta))
        em.append(np.mean(em_sig1))
        em.append(np.mean(em_sig2))
        em.append(np.mean(em_dist))
    
    if cte_config['em_r']:
        emr.append(np.mean(emr_mu))
        emr.append(np.mean(emr_alpha))
        emr.append(np.mean(emr_beta))
        emr.append(np.mean(emr_sig1))
        emr.append(np.mean(emr_sig2))
        emr.append(np.mean(emr_dist))
    
    if cte_config['em_g']:
        emg.append(np.mean(emg_mu))
        emg.append(np.mean(emg_alpha))
        emg.append(np.mean(emg_beta))
        emg.append(np.mean(emg_sig1))
        emg.append(np.mean(emg_sig2))
        emg.append(np.mean(emg_dist))
    
    if cte_config['em_de']:
        emde.append(np.mean(emde_mu))
        emde.append(np.mean(emde_alpha))
        emde.append(np.mean(emde_beta))
        emde.append(np.mean(emde_sig1))
        emde.append(np.mean(emde_sig2))
        emde.append(np.mean(emde_dist))
    
    if cte_config['mle']:
        mle.append(np.mean(mle_mu))
        mle.append(np.mean(mle_alpha))
        mle.append(np.mean(mle_beta))
        mle.append(np.mean(mle_sig1))
        mle.append(np.mean(mle_sig2))
    
    print('LLH:',np.mean(llh))
    
    if cte_config['cte_']:
        print('cte:',cte)
        
        print('log/t:', np.mean(logseq))
        
    if cte_config['cte_g']:
        print('cte_g',cteg)
    
    if cte_config['em_cte']:
        print('em_cte',em)
    
    if cte_config['em_r']:
        print('em_r',emr)
    
    if cte_config['em_g']:
        print('em_g',emg)
    
    if cte_config['em_de']:
        print('em_de',emde)
    
    if cte_config['mle']:
        print('mle',mle)
    
    print('mu',finalmu)
    
    end = timeit.default_timer()
    
    
    if cte_config['variance']:
        if cte_config['cte_']:
            new_sim = cte[:5]
    
        if cte_config['cte_g']:
            new_sim = cteg[:5]
    
        if cte_config['em_cte']:
            new_sim = em[:5]
    
        if cte_config['em_r']:
            new_sim = emr[:5]
    
        if cte_config['em_g']:
            new_sim = emg[:5]
    
        if cte_config['em_de']:
            new_sim = emde[:5]
    
        if cte_config['mle']:
            new_sim = mle[:5]
        
        print(new_sim)
    
        if cte_config['cte_']:
            cte_mu = []
            cte_alpha = []
            cte_beta = []
            cte_sig1 = []
            cte_sig2 = []
            cte_var = []
    
        if cte_config['cte_g']:
            cteg_mu = []
            cteg_alpha = []
            cteg_beta = []
            cteg_sig1 = []
            cteg_sig2 = []
            cteg_var = []
    
        if cte_config['em_cte']:
            em_mu = []
            em_alpha = []
            em_beta = []
            em_sig1 = []
            em_sig2 = []
            em_dist = []
            em_var  = []
    
        if cte_config['em_r']:
            emr_mu = []
            emr_alpha = []
            emr_beta = []
            emr_sig1 = []
            emr_sig2 = []
            emr_dist = []
            emr_var  = []
    
        if cte_config['em_g']:
            emg_mu = []
            emg_alpha = []
            emg_beta = []
            emg_sig1 = []
            emg_sig2 = []
            emg_dist = []
            emg_var  = []
    
        if cte_config['em_de']:
            emde_mu = []
            emde_alpha = []
            emde_beta = []
            emde_sig1 = []
            emde_sig2 = []
            emde_dist = []
            emde_var  = []
    
        if cte_config['mle']: #no_tree_dist
            mle_mu = []
            mle_alpha = []
            mle_beta = []
            mle_sig1 = []
            mle_sig2 = []
            # emde_dist = []
            mle_var  = []
    
        for times in range(cte_config['variance_loop']):
    
            pp = STPPS(new_sim[0],new_sim[1],new_sim[2],new_sim[3],new_sim[4])
            Ini = pp.InitialGenerate(T=[0, 10], S=[[-5, 5], [-5, 5]])
            Ini1 = Ini
            All = [Ini]
            Index = []
            decay = 1
            while True:
                Ini1, PC = pp.OffspringPropagation(Ini,decay)
                if Ini1.shape == (0,3):
                    break
                else:
                    All.append(Ini1)
                    Ini = Ini1
                    Index.append(PC)
    
    
            # pp = STPPS(50,10,5,0.6,0.6)
            # Ini = pp.InitialGenerate(T=[0, 10], S=[[-5, 5], [-5, 5]])
            # Ini1 = Ini
            # All = [Ini]
            # Index = []
            # decay = 1
            # while True:
            #     Ini1, PC = pp.OffspringPropagation(Ini,decay)
            #     if Ini1.shape == (0,3):
            #         break
            #     else:
            #         All.append(Ini1)
            #         Ini = Ini1
            #         Index.append(PC)
            #     #decay += 1
    
    
            #### First Round Clustering
            if cte_config['dataset']:
                data = dataset()
            else:
                data = pp.list2array(All)
            label = pp.dbscan(data,Sampler(data,1,1))
    
    
    
            label = label.reshape((len(label),1))
            idid = np.array([range(1,len(label)+1)]).reshape((len(label),1))
            data = np.concatenate((data,label),axis = 1)
            data = np.concatenate((data,idid),axis = 1) ## add clustering label and id
    
            # print(Index)
    
            epoch = 1
            x_est = []
            y_est = []
            alpha_est = []
            beta_est = []
            x_est_1 = []
            y_est_1 = []
            alpha_est_1 = []
            beta_est_1 = []
            # dist_list = []
            lam_list = []
            lam_list_1 = []
    
    
    
            #### Infer Frame Data
            All_Infer = []
            Index_Infer = []
            cache = []
            delcenter,groupind,i1,i2,ind,bol = pp.del_center1(data)
            All_Infer.append(i1)
            Index_Infer.append(ind)
            All_Infer.append(i2)
    
    
    
    
    
            ### Second Round
    
            starting_generation_index = 1
    
            stage_all1 = np.zeros((1))
            stage_all2 = np.zeros((1))
            stage_index1 = np.zeros((1))
            stage_index2 = np.zeros((1))
            del2 = []
            gIndex = []
            breakcriteria = []
            for i in range(len(delcenter)):
                delcentersub = pp.del2data(delcenter[i])  #to raw data for clustering
                label2 = pp.dbscan(delcentersub,Sampler(delcentersub,1,0))  #clustering
                label2 = label2.reshape((len(label2),1))
                delcentersub = np.concatenate((delcentersub,label2),axis = 1)  ##add clustering label
                Index2 = SearchLabel(delcentersub,data)  ##Search raw id
                Index2 = Index2.reshape((len(Index2),1))
                #print(Index2.shape)
                #print(delcentersub.shape)
                delcentersub = np.concatenate((delcentersub,Index2),axis = 1)  ##add raw id
                delcenter2,groupind2,i12,i22,ind1,ind2,bol = pp.del_center(delcentersub,groupind[i]) #delcenter
                stage_all1 = np.concatenate((stage_all1,i12), axis =0)
                stage_all2 = np.concatenate((stage_all2,i22), axis =0)
                stage_index1 = np.concatenate((stage_index1,ind1), axis =0)
                stage_index2 = np.concatenate((stage_index2,ind2), axis =0)
                ## delcenter2 / groupindex storage
                delcenter2 = np.array(delcenter2)
                groupind2 = np.array(groupind2)
                del2.append(delcenter2)
                gIndex.append(groupind2)
                breakcriteria.append(bol)
            stage_all1 = np.delete(stage_all1,[0],axis = 0)
            stage_all2 = np.delete(stage_all2,[0],axis = 0)
            stage_index1 = np.delete(stage_index1,[0],axis = 0)
            stage_index2 = np.delete(stage_index2,[0],axis = 0)
            All_Infer[-1] = np.concatenate((All_Infer[-1],stage_all1), axis =0)
            All_Infer.append(stage_all2)
            Index_Infer[-1] = np.concatenate((Index_Infer[-1],stage_index1), axis =0)
            Index_Infer.append(stage_index2)
    
    
            ### after second round
            #  if All_Infer[-1].shape[0] == 0:
            #     All_Infer.pop()
            #    Index_Infer.pop()
    
            #### third round
            breakcrit = [False]
            #eps = [1.3,1.1,0.9,0.8,0.6]
            while all(breakcrit) == False:
                genera =0
                stage_all1 = np.zeros((1))
                stage_all2 = np.zeros((1))
                stage_index1 = np.zeros((1))
                stage_index2 = np.zeros((1))
                del3 = []
                gIndex3 = []
                breakcrit = []
                for i in range(len(del2)):
    
                    if len(del2[i].shape) > 1:  ### single or multi array
                        if del2[i].shape[0] == 1:
                            subdel2 = clustering_pipeline(del2[i],data)
                            sub3,group3,i13,i23,ind13,ind23,bol = pp.del_center(subdel2,int(gIndex[i]))
                            stage_all1 = np.concatenate((stage_all1,i13), axis =0)
                            stage_all2 = np.concatenate((stage_all2,i23), axis =0)
                            stage_index1 = np.concatenate((stage_index1,ind13), axis =0)
                            stage_index2 = np.concatenate((stage_index2,ind23), axis =0)
                            sub3 = np.array(sub3)
                            group3 = np.array(group3)
                            del3.append(sub3)
                            gIndex3.append(group3)
                            breakcrit.append(bol)
                        else:
                            for j in range(del2[i].shape[0]):
                                subdel2 = clustering_pipeline(del2[i][j],data)
                                sub3,group3,i13,i23,ind13,ind23,bol = pp.del_center(subdel2,int(gIndex[i][j]))
                                stage_all1 = np.concatenate((stage_all1,i13), axis =0)
                                stage_all2 = np.concatenate((stage_all2,i23), axis =0)
                                stage_index1 = np.concatenate((stage_index1,ind13), axis =0)
                                stage_index2 = np.concatenate((stage_index2,ind23), axis =0)
                                sub3 = np.array(sub3)
                                group3 = np.array(group3)
                                del3.append(sub3)
                                gIndex3.append(group3)
                                breakcrit.append(bol)
                    else:
                        del2_sub = del2[i].tolist()
                        for k in range(len(del2_sub)):
                            subdel2 = clustering_pipeline(del2_sub[k],data)
                            sub3,group3,i13,i23,ind13,ind23,bol = pp.del_center(subdel2,int(gIndex[i][k]))
                            stage_all1 = np.concatenate((stage_all1,i13), axis =0)
                            stage_all2 = np.concatenate((stage_all2,i23), axis =0)
                            stage_index1 = np.concatenate((stage_index1,ind13), axis =0)
                            stage_index2 = np.concatenate((stage_index2,ind23), axis =0)
                            sub3 = np.array(sub3)
                            group3 = np.array(group3)
                            del3.append(sub3)
                            gIndex3.append(group3)
                            breakcrit.append(bol)
    
    
                stage_all1 = np.delete(stage_all1,[0],axis = 0)
                stage_all2 = np.delete(stage_all2,[0],axis = 0)
                stage_index1 = np.delete(stage_index1,[0],axis = 0)
                stage_index2 = np.delete(stage_index2,[0],axis = 0)
                All_Infer[-1] = np.concatenate((All_Infer[-1],stage_all1), axis =0)
                All_Infer.append(stage_all2)
                Index_Infer[-1] = np.concatenate((Index_Infer[-1],stage_index1), axis =0)
                Index_Infer.append(stage_index2)
                del2 = del3
                gIndex = gIndex3
    
            if All_Infer[-1].shape[0] == 0:
              All_Infer.pop()
              Index_Infer.pop()
    
            ### list to array then to list
            ### log likelihood scipy optimise
    
            Alphabet = SortArray(All,data)
            Tree = BuildTree(Alphabet)
            J= Array2Tree(Alphabet,Index,Tree)
    
    
            #Alphabet_Infer = SortArray(All_Infer)
            ## from float to int
            All_Infer_int = []
            for num in All_Infer:
              num = num.astype(int)
              All_Infer_int.append(num)
    
            # Tree_Infer = BuildTree(All_Infer_int)
            # K = Array2Tree_Infer(All_Infer_int,Index_Infer,Tree_Infer)
    
            # dist = zss.simple_distance(J, K, WeirdNode.get_children, WeirdNode.get_label, weird_dist)
    
            # dist_list.append(dist)
    
    
            #             lam = lam + g(param[0],param[1],param[2],param[3],data,j,i)
            #     return lam
            if cte_config['cte_']:
                P_mat = Index2P(data,Index_Infer,All_Infer)
                xa = cte_config['initial_guess']
                result2 = minimize(fun = loglikihood, x0 = xa, method = 'Nelder-Mead', bounds=bounds)
                # print(result2)
                cte_mu.append(np.trace(P_mat))
                cte_alpha.append(result2.x[0])
                cte_beta.append(result2.x[1])
                cte_sig1.append(result2.x[2])
                cte_sig2.append(result2.x[3])
    
    
            if cte_config['cte_g']:
                P_mat = realP(data,All,Index)
                # P_mat = _random_init_P(data.shape[0])
                xa = cte_config['initial_guess']
                result1 = minimize(fun = loglikihood, x0 = xa,method = 'Nelder-Mead', bounds=bounds)
                # print(result1)
                cteg_mu.append(np.trace(P_mat))
                cteg_alpha.append(result1.x[0])
                cteg_beta.append(result1.x[1])
                cteg_sig1.append(result1.x[2])
                cteg_sig2.append(result1.x[3])
    
            if cte_config['em_cte']:
                # P_mat = _random_init_P(data.shape[0])
                P_mat = Index2P(data,Index_Infer,All_Infer)
                # P_mat = realP(data,All,Index)
                xa = cte_config['initial_guess']
                result = minimize(fun = loglikihood, x0 = xa,method = 'Nelder-Mead', bounds=bounds)
                loss = 1
                while loss> cte_config['threshold']:
                    result1 = minimize(fun = loglikihood, x0 = result.x,method = 'Nelder-Mead', bounds=bounds)
                    P_mat = update_P(P_mat,result1.x,data)
                    loss = abs((result.x - result1.x).sum())
                    result = result1
                # print(result)
                # print('111')
                EM = emtree(P_mat, cte_config['sample_size'])
                for item in EM:
                    dist = zss.simple_distance(J, item, WeirdNode.get_children, WeirdNode.get_label, weird_dist)
                    em_dist.append(dist)
                em_mu.append(np.trace(P_mat))
                em_alpha.append(result.x[0])
                em_beta.append(result.x[1])
                em_sig1.append(result.x[2])
                em_sig2.append(result.x[3])
    
            if cte_config['em_r']:
                P_mat = _random_init_P(data.shape[0])
                # P_mat = Index2P(data,Index_Infer,All_Infer)
                # P_mat = realP(data,All,Index)
                xa = cte_config['initial_guess']
                result = minimize(fun = loglikihood, x0 = xa,method = 'Nelder-Mead', bounds=bounds)
                loss = 1
                while loss> cte_config['threshold']:
                    result1 = minimize(fun = loglikihood, x0 = result.x,method = 'Nelder-Mead', bounds=bounds)
                    P_mat = update_P(P_mat,result1.x,data)
                    loss = abs((result.x - result1.x).sum())
                    result = result1
                # print(result)
                # print('111')
                EM = emtree(P_mat, cte_config['sample_size'])
                for item in EM:
                    dist = zss.simple_distance(J, item, WeirdNode.get_children, WeirdNode.get_label, weird_dist)
                    emr_dist.append(dist)
                emr_mu.append(np.trace(P_mat))
                emr_alpha.append(result.x[0])
                emr_beta.append(result.x[1])
                emr_sig1.append(result.x[2])
                emr_sig2.append(result.x[3])
    
    
            if cte_config['em_g']:
                P_mat = realP(data,All,Index)
                xa = cte_config['initial_guess']
                result = minimize(fun = loglikihood, x0 = xa,method = 'Nelder-Mead', bounds=bounds)
                loss = 1
                while loss> cte_config['threshold']:
                    result1 = minimize(fun = loglikihood, x0 = result.x,method = 'Nelder-Mead', bounds=bounds)
                    P_mat = update_P(P_mat,result1.x,data)
                    loss = abs((result.x - result1.x).sum())
                    # print(loss)
                    result = result1
                # print(result)
                # print('111')
                EM = emtree(P_mat, cte_config['sample_size'])
                for item in EM:
                    dist = zss.simple_distance(J, item, WeirdNode.get_children, WeirdNode.get_label, weird_dist)
                    emg_dist.append(dist)
                emg_mu.append(np.trace(P_mat))
                emg_alpha.append(result.x[0])
                emg_beta.append(result.x[1])
                emg_sig1.append(result.x[2])
                emg_sig2.append(result.x[3])
    
    
            if cte_config['em_de']:
                P_mat = _random_init_P(data.shape[0])
                xa = cte_config['initial_guess']
                result = minimize(fun = loglikihood, x0 = xa,method = 'Nelder-Mead', bounds=bounds)
                loss = 1
                while loss> cte_config['threshold']:
                    result1 = minimize(fun = loglikihood, x0 = result.x,method = 'Nelder-Mead', bounds=bounds)
                    P_mat = update_P(P_mat,result1.x,data)
                    P_mat = em_decluster(P_mat)
                    loss = abs((result.x - result1.x).sum())
                    # print(loss)
                    result = result1
                # print(result)
                # print('111')
                EM = emtree(P_mat, cte_config['sample_size'])
                for item in EM:
                    dist = zss.simple_distance(J, item, WeirdNode.get_children, WeirdNode.get_label, weird_dist)
                    emde_dist.append(dist)
                emde_mu.append(np.trace(P_mat))
                emde_alpha.append(result.x[0])
                emde_beta.append(result.x[1])
                emde_sig1.append(result.x[2])
                emde_sig2.append(result.x[3])
    
    
            if cte_config['mle']:
                xa = cte_config['initial_guess_mle']
                if cte_config['mle_nm']:
                    result = minimize(fun = loglikihood_new, x0 = xa, method = 'Nelder-Mead', bounds=bounds_mle)
                if cte_config['mle_cg']:
                    result = minimize(fun = loglikihood_new, x0 = xa, method = 'CG', bounds=bounds_mle)
                if cte_config['mle_newton']:
                    result = minimize(fun = loglikihood_new, x0 = xa, method = 'L-BFGS-B', bounds=bounds_mle)
    
                mle_mu.append(result.x[0])
                mle_alpha.append(result.x[1])
                mle_beta.append(result.x[2])
                mle_sig1.append(result.x[3])
                mle_sig2.append(result.x[4])
    
    
        if cte_config['cte_']:
            cte_var.append(np.std(1/1000 * np.array(cte_mu)))
            cte_var.append(np.std(cte_alpha))
            cte_var.append(np.std(cte_beta))
            cte_var.append(np.std(cte_sig1))
            cte_var.append(np.std(cte_sig2))
            cte_var.append(np.std(dist_list))
    
        if cte_config['cte_g']:
            cteg_var.append(np.std(1/1000 * np.array(cteg_mu)))
            cteg_var.append(np.std(cteg_alpha))
            cteg_var.append(np.std(cteg_beta))
            cteg_var.append(np.std(cteg_sig1))
            cteg_var.append(np.std(cteg_sig2))
    
        if cte_config['em_cte']:
            em_var.append(np.std(1/1000 * np.array(em_mu)))
            em_var.append(np.std(em_alpha))
            em_var.append(np.std(em_beta))
            em_var.append(np.std(em_sig1))
            em_var.append(np.std(em_sig2))
            em_var.append(np.std(em_dist))
    
        if cte_config['em_r']:
            emr_var.append(np.std(1/1000 * np.array(emr_mu)))
            emr_var.append(np.std(emr_alpha))
            emr_var.append(np.std(emr_beta))
            emr_var.append(np.std(emr_sig1))
            emr_var.append(np.std(emr_sig2))
            emr_var.append(np.std(emr_dist))
    
        if cte_config['em_g']:
            emg_var.append(np.std(1/1000 * np.array(emg_mu)))
            emg_var.append(np.std(emg_alpha))
            emg_var.append(np.std(emg_beta))
            emg_var.append(np.std(emg_sig1))
            emg_var.append(np.std(emg_sig2))
            emg_var.append(np.std(emg_dist))
    
        if cte_config['em_de']:
            emde_var.append(np.std(1/1000 * np.array(emde_mu)))
            emde_var.append(np.std(emde_alpha))
            emde_var.append(np.std(emde_beta))
            emde_var.append(np.std(emde_sig1))
            emde_var.append(np.std(emde_sig2))
            emde_var.append(np.std(emde_dist))
    
        if cte_config['mle']:
            mle_var.append(np.std(1/1000 * np.array(mle_mu)))
            mle_var.append(np.std(mle_alpha))
            mle_var.append(np.std(mle_beta))
            mle_var.append(np.std(mle_sig1))
            mle_var.append(np.std(mle_sig2))
    
    
        if cte_config['cte_']:
            print('cte_var:',cte_var)
    
        if cte_config['cte_g']:
            print('cte_g_var',cteg_var)
    
        if cte_config['em_cte']:
            print('em_cte_var',em_var)
    
        if cte_config['em_r']:
            print('em_r_var',emr_var)
    
        if cte_config['em_g']:
            print('em_g_var',emg_var)
    
        if cte_config['em_de']:
            print('em_de_var',emde_var)
    
        if cte_config['mle']:
            print('mle_var',mle_var)