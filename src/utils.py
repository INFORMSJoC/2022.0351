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

# ##### Simulation Setting
# cte_config = dict()

# cte_config['final-mu'] = True

# cte_config['simulation'] = 1 # 1 - 5.1 2- 5.2
# # cte_config['lam'] = [80,30]# = mu * X * Y * T
# cte_config['lam'] = 40# = mu * X * Y * T
# cte_config['alpha'] =10
# cte_config['beta'] = 5
# cte_config['sigx'] = 0.2
# cte_config['sigy'] = 0.2

# cte_config['gamma'] = 0.3

# cte_config['round'] =  11 # default = 1

# ##### Experiment Setup
# cte_config['cte_'] = True  # cte
# cte_config['cte_g'] = False    # cte_groundtruth
# cte_config['em_cte'] = False    # em_cte
# cte_config['em_r'] = False     # em_random
# cte_config['em_g'] = False     # em_groundtruth
# cte_config['mle'] = False  # mle_Nelder-Mead
# cte_config['mle_nm'] = False  # mle_Nelder-Mead
# cte_config['mle_cg'] = False # mle_CG
# cte_config['mle_newton'] = False # mle_L-BFGS-B

# cte_config['em_de'] = False  # em_declustering

# cte_config['variance'] = False #simulation_variance
# cte_config['variance_loop'] = 4 # samples for computing variance

# ##### CTE Parameter
# cte_config['initial_guess'] = [10,5,0.15,0.15]
# cte_config['initial_guess_mle'] = [20,10,5,0.15,0.15] # only for mle
# cte_config['sampler_rate_1'] = 0.075#0.4 #0.3 # 0.20 -20 #0.12
# cte_config['sampler_rate_2'] =0.64 # 0.64 # 0.6 - 20 #0.68

# cte_config['cluster'] = 'd'  #  cte_config['cluster']  d - dbscan , st - stdbscan ,  h - agglomerative , s - som

# cte_config['st-timedecay'] = 0.4  # ST_DBSCAN Para

# cte_config['somsig'] = 0.1  # SOM Para
# cte_config['somlr'] = 0.01  # SOM Para
# cte_config['somx'] = 65 #75  # SOM Para
# cte_config['somepoch'] = 10 #10  # SOM Para


# ##### EM
# cte_config['sample_size'] = 2
# cte_config['em_ini'] = 1
# cte_config['threshold'] = 0.4


# # ##### Dataset
# cte_config['dataset'] = '' #''-simulation 'o'-onlien retail 'e'-earthquake 'c'-citibike '9'-911_crime
# cte_config['select'] = 1000
# cte_config['cte_kde'] = False
# cte_config['em_kde'] = 0
# cte_config['graph'] = False


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

class WeirdNode(object):

    def __init__(self, label):
        self.my_label = label
        self.my_children = list()

    def get_children(node):
        return node.my_children

    def get_label(node):
        return node.my_label

    # def addkid(self, node, before=False):
    #     if before:  self.my_children.insert(0, node)
    #     else:   self.my_children.append(node)
    #     return self
    
    def addkid(self, node, before=False):
        if before:
            self.my_children.insert(0, node)
        else:
            self.my_children.append(node)
        self.my_children.sort(key=lambda x: x.my_label)
        return self


def dataset(item, num):
    
    if item:
        scalert = sklearn.preprocessing.MinMaxScaler((0,10))
        scalerx = sklearn.preprocessing.MinMaxScaler((-5,5))
        scalery = sklearn.preprocessing.MinMaxScaler((-5,5))
        if item == 'e':
            rawdata = pd.read_csv('earth.csv')
            rawdata = rawdata.drop_duplicates(subset=['Time', 'Latitude', 'Longitude'])
            rawdata = rawdata[:num]
            t =  np.array(rawdata['Time']).T
            x = np.array([rawdata['Latitude']]).T
            y = np.array([rawdata['Longitude']]).T
            vect_convert = np.vectorize(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f").timestamp())
            numeric_array = vect_convert(t)
            t = numeric_array.reshape(-1, 1)
            scalert.fit(t)
            t = scalert.transform(t)
            scalerx.fit(x)
            x = scalerx.transform(x)
            scalery.fit(y)
            y = scalery.transform(y)
            # print(np.max(t[-6]))
            txy = np.concatenate((t,x,y),axis = 1)

        if item == '9':
            rawdata = pd.read_csv('911.csv')
            rawdata = rawdata.drop_duplicates(subset=['lat', 'lng', 'timeStamp'])
            rawdata = rawdata[:num]
            t =  np.array(rawdata['timeStamp']).T
            x = np.array([rawdata['lat']]).T
            y = np.array([rawdata['lng']]).T
            t = np.array([datetime.strptime(i, '%Y-%m-%d %H:%M:%S').timestamp() for i in t])
            t = t.reshape(-1, 1)
            scalert.fit(t)
            t = scalert.transform(t)
            scalerx.fit(x)
            x = scalerx.transform(x)
            scalery.fit(y)
            y = scalery.transform(y)
            # print(np.max(t[-6]))
            txy = np.concatenate((t,x,y),axis = 1)

        if item == 'o':
            df = pd.read_excel('Online Retail.xlsx', sheet_name='Online Retail')
            df = df.dropna()
            df = df.drop_duplicates(subset=['Quantity', 'UnitPrice', 'CustomerID', 'Country', 'InvoiceNo','InvoiceDate','StockCode', 'Description'])
            df['Country'],uniques = pd.factorize(df['Country'])
            df_new = df[['Quantity', 'UnitPrice', 'CustomerID', 'Country']]
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            df_new = df_new.dropna()
            pca_result = pca.fit_transform(df_new)
            df['InvoiceDate'] = df['InvoiceDate'].apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S').timestamp())
            df = df.drop(['StockCode', 'Description', 'Quantity', 'UnitPrice', 'CustomerID', 'Country'], axis=1)
            df_pca = pd.DataFrame(data = pca_result, columns = ['x', 'y'])
            df = pd.concat([df, df_pca], axis=1)
            df = df.groupby('InvoiceNo').mean().reset_index()
            df = df.drop(['InvoiceNo'], axis=1)
            
            x = np.array([df['x']]).T
            y = np.array([df['y']]).T
            t = np.array([df['InvoiceDate']]).T
            scalert = sklearn.preprocessing.MinMaxScaler((0,10))
            scalerx = sklearn.preprocessing.MinMaxScaler((-5,5))
            scalery = sklearn.preprocessing.MinMaxScaler((-5,5))
            x = x[1000:num+1000]
            y = y[1000:num+1000]
            t = t[1000:num+1000]
            scalert.fit(t)
            t = scalert.transform(t)
            scalerx.fit(x)
            x = scalerx.transform(x)
            scalery.fit(y)
            y = scalery.transform(y)
            # print(np.max(x))]
            txy = np.concatenate((t,x,y),axis = 1)

        if item == 'c':
            rawdata = pd.read_csv('citibike.csv')
            rawdata = rawdata[:num]
            rawdata = rawdata.drop_duplicates(subset=['starttime', 'start station latitude', 'start station longitude'])
            # rawdata.drop_duplicates('start station latitude')
            # rawdata.drop_duplicates('start station longitude')
            t =  np.array([rawdata['starttime']]).T
            for i in range(len(t)):
                k = pd.to_datetime(t[i,0]).timestamp()
                t[i,0] = k
                # print(k)
            scalert.fit(t)
            t = scalert.transform(t)
            x = np.array([rawdata['start station latitude']]).T
            scalerx.fit(x)
            x = scalerx.transform(x)
            y = np.array([rawdata['start station longitude']]).T
            scalery.fit(y)
            y = scalery.transform(y)
            # print(np.max(x))]
            txy = np.concatenate((t,x,y),axis = 1)
            # print(x.shape)
        return txy
    else:
        pass

    # return x,y



class STPPS(object):





    def __init__(self,lam,alpha,beta,sigma1,sigma2,config_cte):
        self.lam = lam
        self.alpha = alpha
        self.beta = beta
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.cte_config = config_cte
        #self.eps = eps


    def InitialGenerate(self, T=[0, 10], S=[[-5, 5], [-5, 5]]):
        _S     = [T] + S
        N      = np.random.poisson(size=1, lam = self.lam)
        points = [ np.random.uniform(_S[i][0], _S[i][1], N) for i in range(len(_S)) ]
        points = np.array(points).transpose()
        points = points[points[:, 0].argsort()]
        return points


    def OffspringPropagation(self,points,decay,T=[0, 10]):
        N0 = len(points)
        PCindex = np.zeros((1))
        i = 1
        mean = [0,0]
        cov = [[(self.sigma1**2)*(1/decay),0],[0,(self.sigma2**2)*(1/decay)]]
        offspring = np.array([[0,0,0]])
        while i <= N0:
            M = np.random.poisson(size=1, lam = (2* math.pi * self.sigma1 * self.sigma2 * self.alpha)/self.beta)
            if M == 0:
                i += 1
            else:
                j = 1
                dt = 0
                dt1 = np.random.exponential(scale = (1/self.beta) *(1/decay), size = M)
                dt1 = dt1[dt1.argsort()]
                while j <= M:
                    #dt1 = np.random.exponential(scale = 0.2*(1/decay), size = M)
                    #dt1 = dt1[dt1.argsort()]
                    dxdy = np.random.multivariate_normal(mean,cov,1)
                    dt = dt1[j-1]
                    jj = np.array([np.append(dt,dxdy)])
                    jj = jj + points[i-1]
                    if jj[0,0] <= T[1]:
                        if abs(jj[0,1]) <= 5:
                            if abs(jj[0,2]) <=5:
                                offspring = np.concatenate((offspring,jj),axis = 0)
                                PCindex = np.concatenate((PCindex,np.array([i])),axis = 0)
                    j += 1
                i += 1
        offspring = np.delete(offspring,[0],axis = 0)
        PCindex = np.delete(PCindex,[0],axis = 0)
        return offspring, PCindex


    def list2array(self,list):
        array = np.array(list)
        ini = np.array([[0,0,0]])
        for i in array:
            ini = np.concatenate((ini,i),axis = 0)
        end = np.delete(ini,[0],axis = 0)
        endd = end
        endd = endd[endd[:, 0].argsort()]
        return endd


    def Off2Plt(self,array):
        plt.scatter(array[:,1],array[:,2], marker = 'o')
        plt.show()


    def dbscan(self,data,epsilon):
        if self.cte_config['cluster'] == 'd':
            y_pred = sklearn.cluster.DBSCAN(eps = epsilon, min_samples = 1, metric = 'l1').fit_predict(data)
        elif self.cte_config['cluster'] == 'st':
            y_pred = ST_DBSCAN(eps1 = epsilon[0], eps2 = epsilon[1], min_samples = 1).fit(data)
            y_pred = y_pred.labels
        elif self.cte_config['cluster'] == 'h':
            y_pred = AgglomerativeClustering(n_clusters = None, distance_threshold=epsilon, compute_full_tree= True).fit_predict(data)
        else:  ## SOM
            som = MiniSom(self.cte_config['somx'], self.cte_config['somx'], 3, sigma=self.cte_config['somsig'], learning_rate= self.cte_config['somlr'])# initialization of 6x6 SOM
            som.train(data, self.cte_config['somepoch'])
            winner_coordinates = np.array([som.winner(x) for x in data]).T
            y_pred = np.ravel_multi_index(winner_coordinates, (self.cte_config['somx'],self.cte_config['somx']))
        # plt.scatter(data[:, 1], data[:, 2], c=y_pred)
        # plt.show()
        return y_pred



        # if cte_config['clustering'] == 'st':
        #   y_pred = ST_DBSCAN(eps1 = epsilon1, eps2 = epsilon2, min_samples = 1).fit(data)
        # if cte_config['clustering'] == 'h':
        #   y_pred = AgglomerativeClustering(n_clusters = None, distance_threshold=epsilon, compute_full_tree= True).fit_predict(data)
        # if cte_config['clustering'] == 's':
        #   som = MiniSom(10, 10, 4, sigma=0.3, learning_rate=0.5).train(data,10) # initialization of 6x6 SOM
        #   winner_coordinates = np.array([som.winner(x) for x in data]).T
        #   y_pred = np.ravel_multi_index(winner_coordinates, (10,10))







    def add_id(self,data):
        dataid = np.array(range(1,len(data)))
        dataid = dataid.reshape((len(dataid),1))
        data = np.concatenate((data,dataid),axis = 1)
        return data


    def del2data(self,data):
        data1 = data[:,:3]
        return data1


    def del_center1(self,data): ## data has id[4] and groupid[3]
        #All = []   ## Current generation item
        Index1 = np.zeros((1)) ## parent index
        Index2 = np.zeros((1)) ## child index
        Index = np.zeros((1))
        grouplist = []
        groupindex = []
        for i in range(int(np.max(data[:,3])+1)): #cover all the category
            labelsum = 0
            group = np.array([[0,0,0,0,0]])
            for j in range(len(data)):
                if data[j,3] == i:
                    labelsum += 1
                    group = np.concatenate((group,np.array([data[j]])),axis = 0)
                j += 1
            group = group[group[:,0].argsort()]  #Sorting by time order
            group = np.delete(group,[0],axis = 0) #delete 0
            if labelsum > 2:
                groupindex.append(group[0][-1]) #recording the parent for all
                Index1 = np.concatenate((Index1,np.array([group[0][-1]])),axis = 0) #recording the parent into current generation
                group = np.delete(group,[0],axis = 0)  #Delete the parent
                grouplist.append(group)  #recording the rest
            elif labelsum == 2:
                Index1 = np.concatenate((Index1,np.array([group[0][-1]])),axis = 0)
                Index2 = np.concatenate((Index2,np.array([group[1][-1]])),axis = 0)
                Index = np.concatenate((Index,np.array([group[0][-1]])),axis = 0)
            elif labelsum == 1:
                Index1 = np.concatenate((Index1,np.array([group[0][-1]])),axis = 0)
        Index1 = np.delete(Index1,[0],axis = 0) ##delete 0
        Index2 = np.delete(Index2,[0],axis = 0) ##delete 0
        Index = np.delete(Index,[0],axis = 0) ##delete 0
        Boolean = not grouplist
        return grouplist,groupindex,Index1,Index2,Index,Boolean


    def Mergelist(self,data):
        if len(data) > 2:
            new = np.concatenate(data[0],data[1],axis = 0)
            if len(data) >3:
                for i in range(2,len(data)):
                    new = np.concatenate(new,data[i],axis = 0)
        return new



    def del_center(self,data,groupind): ## data has id[4] and groupid[3]
        #All = []   ## Current generation item
        Index1 = np.zeros((1)) ## parent index
        Index2 = np.zeros((1)) ## child index
        RealIndex1 = np.zeros((1))
        RealIndex2 = np.zeros((1))
        grouplist = []
        groupindex = []
        for i in range(int(np.max(data[:,3])+1)): #cover all the category
            labelsum = 0
            group = np.array([[0,0,0,0,0]])
            for j in range(len(data)):
                if data[j,3] == i:
                    labelsum += 1
                    group = np.concatenate((group,np.array([data[j]])),axis = 0)
                j += 1
            group = group[group[:,0].argsort()]  #Sorting by time order
            group = np.delete(group,[0],axis = 0) #delete 0
            if labelsum > 2:
                groupindex.append(group[0][-1]) #recording the parent for all
                Index1 = np.concatenate((Index1,np.array([group[0][-1]])),axis = 0) #recording the parent into current generation
                group = np.delete(group,[0],axis = 0)  #Delete the parent
                grouplist.append(group)  #recording the rest
                RealIndex1 = np.concatenate((RealIndex1,np.array([groupind])),axis = 0)
            elif labelsum == 2:
                Index1 = np.concatenate((Index1,np.array([group[0][-1]])),axis = 0)
                Index2 = np.concatenate((Index2,np.array([group[1][-1]])),axis = 0)
                RealIndex2 = np.concatenate((RealIndex2,np.array([group[0][-1]])),axis = 0)
                RealIndex1 = np.concatenate((RealIndex1,np.array([groupind])),axis = 0)
            elif labelsum == 1:
                Index1 = np.concatenate((Index1,np.array([group[0][-1]])),axis = 0)
                RealIndex1 = np.concatenate((RealIndex1,np.array([groupind])),axis = 0)
        Index1 = np.delete(Index1,[0],axis = 0) ##delete 0
        Index2 = np.delete(Index2,[0],axis = 0) ##delete 0
        RealIndex1 = np.delete(RealIndex1,[0],axis = 0) ##delete 0
        RealIndex2 = np.delete(RealIndex2,[0],axis = 0) ##delete 0
        Boolean = not grouplist
        return grouplist,groupindex,Index1,Index2,RealIndex1,RealIndex2,Boolean





class STPPS2(object):





    def __init__(self,lam,alpha,beta,gamma,sigma1,sigma2, config_cte):
        self.lam = lam
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.cte_config = config_cte
        #self.eps = eps


    def InitialGenerate(self, T=[0, 10], S=[[-5, 5], [-5, 5]]):
        _S1     = [[T[0],int(T[1]/2)]] + S
        _S2     = [[int(T[1]/2),T[1]]] + S
        N1      = np.random.poisson(size=1, lam = self.lam[0])
        N2      = np.random.poisson(size=1, lam = self.lam[1])
        points1 = [ np.random.uniform(_S1[i][0], _S1[i][1], N1) for i in range(len(_S1)) ]
        points2 = [ np.random.uniform(_S2[i][0], _S2[i][1], N2) for i in range(len(_S2)) ]
        points1 = np.array(points1).transpose()
        points2 = np.array(points2).transpose()
        points = np.concatenate((points1,points2),axis=0)
        points = points[points[:, 0].argsort()]
        return points


    def OffspringPropagation(self,points,decay,T=[0, 10]):
        N0 = len(points)
        PCindex = np.zeros((1))
        i = 1
        # mean = [0,0]
        # cov = [[(self.sigma1**2)*(1/decay),0],[0,(self.sigma2**2)*(1/decay)]]
        offspring = np.array([[0,0,0]])
        while i <= N0:
            m = np.random.poisson(size=1, lam = 1)
            M = m[0]
            # print(M)
            # M = np.random.poisson(size=1, lam = (2* math.pi * self.sigma1 * self.sigma2 * self.alpha)/self.beta)
            if M == 0:
                i += 1
            else:
                j = 1
                ti = points[i-1][0]
                xi = points[i-1][1]
                yi = points[i-1][2]
                dt = np.array([np.random.uniform(ti ,T[1] ,size = M)])
                dx = np.array([np.random.uniform(-5 ,5 ,size = M)])
                dy = np.array([np.random.uniform(-5 ,5 ,size = M)])
                offsp = np.concatenate((dt.T,dx.T,dy.T),axis = 1)
                # print(offsp)
                prob = [np.exp( -self.gamma * np.linalg.norm([dt[0,k] - ti, dx[0,k] -xi, dy[0,k] - yi])) for k in range(M)]
                accept = [np.random.binomial(1,prob[k]) for k in range(M)]
                # dt1 = np.random.exponential(scale = (1/self.beta) *(1/decay), size = M)
                # dt1 = dt1[dt1.argsort()]
                while j <= M:
                    #dt1 = np.random.exponential(scale = 0.2*(1/decay), size = M)
                    #dt1 = dt1[dt1.argsort()]
                    if accept[j-1] > 0:
                        #print('1')
                        offspring = np.concatenate((offspring,np.array([offsp[j-1]])),axis = 0)
                        PCindex = np.concatenate((PCindex,np.array([i])),axis = 0)

                    # prob = [np.exp( -gamma * np.linalg.norm([dt[k] - ti, dx[k] -xi, dy[k] - yi])) for k in range(M)]
                    # accept = [np.random.binomial(1,prob[k]) for k in range(M)]

                    # dxdy = np.random.multivariate_normal(mean,cov,1)
                    # dt = dt1[j-1]
                    # jj = np.array([np.append(dt,dxdy)])
                    # jj = jj + points[i-1]
                    # if jj[0,0] <= T[1]:
                    #     if abs(jj[0,1]) <= 5:
                    #         if abs(jj[0,2]) <=5:
                    #             offspring = np.concatenate((offspring,jj),axis = 0)
                    #             PCindex = np.concatenate((PCindex,np.array([i])),axis = 0)
                    j += 1
                i += 1
        offspring = np.delete(offspring,[0],axis = 0)
        PCindex = np.delete(PCindex,[0],axis = 0)
        return offspring, PCindex


    def list2array(self,list):
        array = np.array(list)
        ini = np.array([[0,0,0]])
        for i in array:
            ini = np.concatenate((ini,i),axis = 0)
        end = np.delete(ini,[0],axis = 0)
        endd = end
        endd = endd[endd[:, 0].argsort()]
        return endd


    def Off2Plt(self,array):
        plt.scatter(array[:,1],array[:,2], marker = 'o')
        plt.show()


    def dbscan(self,data,epsilon):
        if self.cte_config['cluster'] == 'd':
            y_pred = sklearn.cluster.DBSCAN(eps = epsilon, min_samples = 1, metric = 'l1').fit_predict(data)
        elif self.cte_config['cluster'] == 'st':
            y_pred = ST_DBSCAN(eps1 = epsilon[0], eps2 = epsilon[1], min_samples = 1).fit(data)
            y_pred = y_pred.labels
        elif self.cte_config['cluster'] == 'h':
            y_pred = AgglomerativeClustering(n_clusters = None, distance_threshold=epsilon, compute_full_tree= True).fit_predict(data)
        else:  ## SOM
            som = MiniSom(self.cte_config['somx'], self.cte_config['somx'], 3, sigma=self.cte_config['somsig'], learning_rate= self.cte_config['somlr'])# initialization of 6x6 SOM
            som.train(data, self.cte_config['somepoch'])
            winner_coordinates = np.array([som.winner(x) for x in data]).T
            y_pred = np.ravel_multi_index(winner_coordinates, (self.cte_config['somx'],self.cte_config['somx']))
        # plt.scatter(data[:, 1], data[:, 2], c=y_pred)
        # plt.show()
        return y_pred



        # if cte_config['clustering'] == 'st':
        #   y_pred = ST_DBSCAN(eps1 = epsilon1, eps2 = epsilon2, min_samples = 1).fit(data)
        # if cte_config['clustering'] == 'h':
        #   y_pred = AgglomerativeClustering(n_clusters = None, distance_threshold=epsilon, compute_full_tree= True).fit_predict(data)
        # if cte_config['clustering'] == 's':
        #   som = MiniSom(10, 10, 4, sigma=0.3, learning_rate=0.5).train(data,10) # initialization of 6x6 SOM
        #   winner_coordinates = np.array([som.winner(x) for x in data]).T
        #   y_pred = np.ravel_multi_index(winner_coordinates, (10,10))







    def add_id(self,data):
        dataid = np.array(range(1,len(data)))
        dataid = dataid.reshape((len(dataid),1))
        data = np.concatenate((data,dataid),axis = 1)
        return data


    def del2data(self,data):
        data1 = data[:,:3]
        return data1


    def del_center1(self,data): ## data has id[4] and groupid[3]
        #All = []   ## Current generation item
        Index1 = np.zeros((1)) ## parent index
        Index2 = np.zeros((1)) ## child index
        Index = np.zeros((1))
        grouplist = []
        groupindex = []
        for i in range(int(np.max(data[:,3])+1)): #cover all the category
            labelsum = 0
            group = np.array([[0,0,0,0,0]])
            for j in range(len(data)):
                if data[j,3] == i:
                    labelsum += 1
                    group = np.concatenate((group,np.array([data[j]])),axis = 0)
                j += 1
            group = group[group[:,0].argsort()]  #Sorting by time order
            group = np.delete(group,[0],axis = 0) #delete 0
            if labelsum > 2:
                groupindex.append(group[0][-1]) #recording the parent for all
                Index1 = np.concatenate((Index1,np.array([group[0][-1]])),axis = 0) #recording the parent into current generation
                group = np.delete(group,[0],axis = 0)  #Delete the parent
                grouplist.append(group)  #recording the rest
            elif labelsum == 2:
                Index1 = np.concatenate((Index1,np.array([group[0][-1]])),axis = 0)
                Index2 = np.concatenate((Index2,np.array([group[1][-1]])),axis = 0)
                Index = np.concatenate((Index,np.array([group[0][-1]])),axis = 0)
            elif labelsum == 1:
                Index1 = np.concatenate((Index1,np.array([group[0][-1]])),axis = 0)
        Index1 = np.delete(Index1,[0],axis = 0) ##delete 0
        Index2 = np.delete(Index2,[0],axis = 0) ##delete 0
        Index = np.delete(Index,[0],axis = 0) ##delete 0
        Boolean = not grouplist
        return grouplist,groupindex,Index1,Index2,Index,Boolean


    def Mergelist(self,data):
        if len(data) > 2:
            new = np.concatenate(data[0],data[1],axis = 0)
            if len(data) >3:
                for i in range(2,len(data)):
                    new = np.concatenate(new,data[i],axis = 0)
        return new



    def del_center(self,data,groupind): ## data has id[4] and groupid[3]
        #All = []   ## Current generation item
        Index1 = np.zeros((1)) ## parent index
        Index2 = np.zeros((1)) ## child index
        RealIndex1 = np.zeros((1))
        RealIndex2 = np.zeros((1))
        grouplist = []
        groupindex = []
        for i in range(int(np.max(data[:,3])+1)): #cover all the category
            labelsum = 0
            group = np.array([[0,0,0,0,0]])
            for j in range(len(data)):
                if data[j,3] == i:
                    labelsum += 1
                    group = np.concatenate((group,np.array([data[j]])),axis = 0)
                j += 1
            group = group[group[:,0].argsort()]  #Sorting by time order
            group = np.delete(group,[0],axis = 0) #delete 0
            if labelsum > 2:
                groupindex.append(group[0][-1]) #recording the parent for all
                Index1 = np.concatenate((Index1,np.array([group[0][-1]])),axis = 0) #recording the parent into current generation
                group = np.delete(group,[0],axis = 0)  #Delete the parent
                grouplist.append(group)  #recording the rest
                RealIndex1 = np.concatenate((RealIndex1,np.array([groupind])),axis = 0)
            elif labelsum == 2:
                Index1 = np.concatenate((Index1,np.array([group[0][-1]])),axis = 0)
                Index2 = np.concatenate((Index2,np.array([group[1][-1]])),axis = 0)
                RealIndex2 = np.concatenate((RealIndex2,np.array([group[0][-1]])),axis = 0)
                RealIndex1 = np.concatenate((RealIndex1,np.array([groupind])),axis = 0)
            elif labelsum == 1:
                Index1 = np.concatenate((Index1,np.array([group[0][-1]])),axis = 0)
                RealIndex1 = np.concatenate((RealIndex1,np.array([groupind])),axis = 0)
        Index1 = np.delete(Index1,[0],axis = 0) ##delete 0
        Index2 = np.delete(Index2,[0],axis = 0) ##delete 0
        RealIndex1 = np.delete(RealIndex1,[0],axis = 0) ##delete 0
        RealIndex2 = np.delete(RealIndex2,[0],axis = 0) ##delete 0
        Boolean = not grouplist
        return grouplist,groupindex,Index1,Index2,RealIndex1,RealIndex2,Boolean



class STPPS3(object):





    def __init__(self,lam,alpha,beta,sigma1,sigma2,config_cte):
        self.lam = lam
        self.alpha = alpha
        self.beta = beta
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.cte_config = config_cte
        #self.eps = eps


    def InitialGenerate(self, T=[0, 10], S=[[-5, 5], [-5, 5]]):
        _S     = [T] + S
        N      = np.random.poisson(size=1, lam = self.lam)
        points = [ np.random.uniform(_S[i][0], _S[i][1], N) for i in range(len(_S)) ]
        points = np.array(points).transpose()
        points = points[points[:, 0].argsort()]
        return points


    def OffspringPropagation(self,points,decay,T=[0, 10], alpha = 0, beta = 0, sigma1 = 0, sigma2 = 0):
        N0 = len(points)
        PCindex = np.zeros((1))
        i = 1
        mean = [0,0]
        cov = [[(sigma1**2)*(1/decay),0],[0,(sigma2**2)*(1/decay)]]
        offspring = np.array([[0,0,0]])
        while i <= N0:
            M = np.random.poisson(size=1, lam = (2* math.pi * sigma1 * sigma2 * alpha)/beta)
            if M == 0:
                i += 1
            else:
                j = 1
                dt = 0
                dt1 = np.random.exponential(scale = (1/beta) *(1/decay), size = M)
                dt1 = dt1[dt1.argsort()]
                while j <= M:
                    #dt1 = np.random.exponential(scale = 0.2*(1/decay), size = M)
                    #dt1 = dt1[dt1.argsort()]
                    dxdy = np.random.multivariate_normal(mean,cov,1)
                    dt = dt1[j-1]
                    jj = np.array([np.append(dt,dxdy)])
                    jj = jj + points[i-1]
                    if jj[0,0] <= T[1]:
                        if abs(jj[0,1]) <= 5:
                            if abs(jj[0,2]) <=5:
                                offspring = np.concatenate((offspring,jj),axis = 0)
                                PCindex = np.concatenate((PCindex,np.array([i])),axis = 0)
                    j += 1
                i += 1
        offspring = np.delete(offspring,[0],axis = 0)
        PCindex = np.delete(PCindex,[0],axis = 0)
        return offspring, PCindex


    def list2array(self,list):
        array = np.array(list)
        ini = np.array([[0,0,0]])
        for i in array:
            ini = np.concatenate((ini,i),axis = 0)
        end = np.delete(ini,[0],axis = 0)
        endd = end
        endd = endd[endd[:, 0].argsort()]
        return endd


    def Off2Plt(self,array):
        plt.scatter(array[:,1],array[:,2], marker = 'o')
        plt.show()


    def dbscan(self,data,epsilon):
        if self.cte_config['cluster'] == 'd':
            y_pred = sklearn.cluster.DBSCAN(eps = epsilon, min_samples = 1, metric = 'l1').fit_predict(data)
        elif self.cte_config['cluster'] == 'st':
            y_pred = ST_DBSCAN(eps1 = epsilon[0], eps2 = epsilon[1], min_samples = 1).fit(data)
            y_pred = y_pred.labels
        elif self.cte_config['cluster'] == 'h':
            y_pred = AgglomerativeClustering(n_clusters = None, distance_threshold=epsilon, compute_full_tree= True).fit_predict(data)
        else:  ## SOM
            som = MiniSom(self.cte_config['somx'], self.cte_config['somx'], 3, sigma=self.cte_config['somsig'], learning_rate= self.cte_config['somlr'])# initialization of 6x6 SOM
            som.train(data, self.cte_config['somepoch'])
            winner_coordinates = np.array([som.winner(x) for x in data]).T
            y_pred = np.ravel_multi_index(winner_coordinates, (self.cte_config['somx'],self.cte_config['somx']))
        # plt.scatter(data[:, 1], data[:, 2], c=y_pred)
        # plt.show()
        return y_pred



        # if cte_config['clustering'] == 'st':
        #   y_pred = ST_DBSCAN(eps1 = epsilon1, eps2 = epsilon2, min_samples = 1).fit(data)
        # if cte_config['clustering'] == 'h':
        #   y_pred = AgglomerativeClustering(n_clusters = None, distance_threshold=epsilon, compute_full_tree= True).fit_predict(data)
        # if cte_config['clustering'] == 's':
        #   som = MiniSom(10, 10, 4, sigma=0.3, learning_rate=0.5).train(data,10) # initialization of 6x6 SOM
        #   winner_coordinates = np.array([som.winner(x) for x in data]).T
        #   y_pred = np.ravel_multi_index(winner_coordinates, (10,10))







    def add_id(self,data):
        dataid = np.array(range(1,len(data)))
        dataid = dataid.reshape((len(dataid),1))
        data = np.concatenate((data,dataid),axis = 1)
        return data


    def del2data(self,data):
        data1 = data[:,:3]
        return data1


    def del_center1(self,data): ## data has id[4] and groupid[3]
        #All = []   ## Current generation item
        Index1 = np.zeros((1)) ## parent index
        Index2 = np.zeros((1)) ## child index
        Index = np.zeros((1))
        grouplist = []
        groupindex = []
        for i in range(int(np.max(data[:,3])+1)): #cover all the category
            labelsum = 0
            group = np.array([[0,0,0,0,0]])
            for j in range(len(data)):
                if data[j,3] == i:
                    labelsum += 1
                    group = np.concatenate((group,np.array([data[j]])),axis = 0)
                j += 1
            group = group[group[:,0].argsort()]  #Sorting by time order
            group = np.delete(group,[0],axis = 0) #delete 0
            if labelsum > 2:
                groupindex.append(group[0][-1]) #recording the parent for all
                Index1 = np.concatenate((Index1,np.array([group[0][-1]])),axis = 0) #recording the parent into current generation
                group = np.delete(group,[0],axis = 0)  #Delete the parent
                grouplist.append(group)  #recording the rest
            elif labelsum == 2:
                Index1 = np.concatenate((Index1,np.array([group[0][-1]])),axis = 0)
                Index2 = np.concatenate((Index2,np.array([group[1][-1]])),axis = 0)
                Index = np.concatenate((Index,np.array([group[0][-1]])),axis = 0)
            elif labelsum == 1:
                Index1 = np.concatenate((Index1,np.array([group[0][-1]])),axis = 0)
        Index1 = np.delete(Index1,[0],axis = 0) ##delete 0
        Index2 = np.delete(Index2,[0],axis = 0) ##delete 0
        Index = np.delete(Index,[0],axis = 0) ##delete 0
        Boolean = not grouplist
        return grouplist,groupindex,Index1,Index2,Index,Boolean


    def Mergelist(self,data):
        if len(data) > 2:
            new = np.concatenate(data[0],data[1],axis = 0)
            if len(data) >3:
                for i in range(2,len(data)):
                    new = np.concatenate(new,data[i],axis = 0)
        return new



    def del_center(self,data,groupind): ## data has id[4] and groupid[3]
        #All = []   ## Current generation item
        Index1 = np.zeros((1)) ## parent index
        Index2 = np.zeros((1)) ## child index
        RealIndex1 = np.zeros((1))
        RealIndex2 = np.zeros((1))
        grouplist = []
        groupindex = []
        for i in range(int(np.max(data[:,3])+1)): #cover all the category
            labelsum = 0
            group = np.array([[0,0,0,0,0]])
            for j in range(len(data)):
                if data[j,3] == i:
                    labelsum += 1
                    group = np.concatenate((group,np.array([data[j]])),axis = 0)
                j += 1
            group = group[group[:,0].argsort()]  #Sorting by time order
            group = np.delete(group,[0],axis = 0) #delete 0
            if labelsum > 2:
                groupindex.append(group[0][-1]) #recording the parent for all
                Index1 = np.concatenate((Index1,np.array([group[0][-1]])),axis = 0) #recording the parent into current generation
                group = np.delete(group,[0],axis = 0)  #Delete the parent
                grouplist.append(group)  #recording the rest
                RealIndex1 = np.concatenate((RealIndex1,np.array([groupind])),axis = 0)
            elif labelsum == 2:
                Index1 = np.concatenate((Index1,np.array([group[0][-1]])),axis = 0)
                Index2 = np.concatenate((Index2,np.array([group[1][-1]])),axis = 0)
                RealIndex2 = np.concatenate((RealIndex2,np.array([group[0][-1]])),axis = 0)
                RealIndex1 = np.concatenate((RealIndex1,np.array([groupind])),axis = 0)
            elif labelsum == 1:
                Index1 = np.concatenate((Index1,np.array([group[0][-1]])),axis = 0)
                RealIndex1 = np.concatenate((RealIndex1,np.array([groupind])),axis = 0)
        Index1 = np.delete(Index1,[0],axis = 0) ##delete 0
        Index2 = np.delete(Index2,[0],axis = 0) ##delete 0
        RealIndex1 = np.delete(RealIndex1,[0],axis = 0) ##delete 0
        RealIndex2 = np.delete(RealIndex2,[0],axis = 0) ##delete 0
        Boolean = not grouplist
        return grouplist,groupindex,Index1,Index2,RealIndex1,RealIndex2,Boolean




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

    # print(data_len)
    # print(epsilon)
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

