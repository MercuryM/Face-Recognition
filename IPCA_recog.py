import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from scipy.linalg import *


from Split import split_data
from Split_1 import split_4_subsets

def cal_data(dataset):
    data_num = len(dataset[1,:])
    mean_data = np.reshape((np.sum(dataset, axis=1))*(1/data_num),(2576,1))
    data_mat = dataset - mean_data
    data_cov = (1/data_num)*(data_mat.dot(data_mat.T))
    data_cov_1 = (1/data_num)*((data_mat.T).dot(data_mat))   
    eigvals, LD_eigvecs = np.linalg.eig(data_cov_1)
    
    idx_L = eigvals.argsort()[::-1]
    eigvals = eigvals[idx_L]
    LD_eigvecs = LD_eigvecs[:,idx_L]

    eigvecs = data_mat.dot(LD_eigvecs)/np.linalg.norm(data_mat.dot(LD_eigvecs),axis=0)
#    eigvecs = eigvecs[:,0:len(eigvecs)]
       
    return {'mean_data': mean_data, 'data_mat': data_mat,'data_cov':data_cov,'vals':eigvals,'vecs':eigvecs}
        

def combine_data(dataset1, dataset2, mean1, cov1, vecs1,M):
    N_1 = len(dataset1[1,:])
    N_2 = len(dataset2[1,:])
    mean2,cov2,eigvecs2 = cal_data(dataset2)['mean_data'],cal_data(dataset2)['data_cov'],cal_data(dataset2)['vecs']
    N_3 = N_1 + N_2
    comb_mean = (N_1 * mean1 + N_2 * mean2)/N_3
    comb_cov = (N_1/N_3)*cov1 + (N_2/N_3)*cov2 + ((N_1*N_2)/(N_3**2))*(mean1-mean2)*((mean1-mean2).T)
    h = np.concatenate((vecs1,eigvecs2[:,0:M],(mean1-mean2)),axis = 1)
    orth_comb_cov = orth(h)
    small_cov = np.matmul  (np.matmul(orth_comb_cov.T,comb_cov),orth_comb_cov)
    new_eigvals, eigvecs_R = np.linalg.eig(small_cov)
    new_eigvecs = np.matmul(orth_comb_cov,eigvecs_R)   
    return {'mean': comb_mean,'cov': comb_cov,'vals':new_eigvals,'vecs':new_eigvecs}
    



data = split_data()
training_X, training_l = data['train']
D, training_num = training_X.shape
testing_X, testing_l = data['test']
I, K = testing_X.shape

data_1 = split_4_subsets(training_X, training_l)

subset1_X, subset1_l = data_1['subset1']
subset2_X, subset2_l = data_1['subset2']
subset3_X, subset3_l = data_1['subset3']
subset4_X, subset4_l = data_1['subset4']
subset_X = np.concatenate((subset1_X,subset2_X,subset3_X,subset4_X),axis = 1)
subset_l = np.concatenate((subset1_l,subset2_l,subset3_l,subset4_l),axis = 1)

M = [13,26,53,104-1]
ipca_counter = []
ipca_acc = []
ipca_t = []                
for i in range(4):
    counter = []
    process_time_1 = []
    counter_1 = []
    acc_1 = []
    for j in range(4):
        
        if j == 0:
            t_start = time.time()
            data1 = cal_data(subset1_X)
            t_finish = time.time() - t_start
            process_time_1.append(t_finish)
            mean1,mat1,cov1,eigvals1,eigvecs1 = data1['mean_data'],data1['data_mat'],data1['data_cov'],data1['vals'],data1['vecs']
            eigvecs1 = eigvecs1[:,0:M[i]]
            W_n = (mat1.T).dot(eigvecs1)
            testing_mat = testing_X-mean1
            W = (testing_mat.T).dot(eigvecs1)
            err_mat = np.full((1,104),1000)

            diff_mat = np.full((len(W_n[:,1]),M[i]),1000)
            
            for k in range(len(W[:,1])):
                diff_mat[:,:] = W[k,:] - W_n
                err = np.linalg.norm(diff_mat,axis = 1)
                err_mat[0,k] = err.argmin()
            ID_real = np.reshape(testing_l,(104,))    
#            for m in range(len(err_mat[:,1])):
            ID_recog = np.reshape(subset1_l[:,err_mat[0,:]].astype(int),(104,))
            right = 0        
            for n in range(len(ID_real)):
                if(ID_real[n]==ID_recog[n]):
                    right = right+1
                        
            acc_1.append(right/104)     
        else:
            t_start = time.time()
            combine = combine_data(subset_X[:,0:(104*j)],subset_X[:,(104*j):(104*(j+1))], mean1, cov1, eigvecs1, M[i])
            t_finish = time.time() - t_start
            process_time_1.append(t_finish)
            
            mean1 = combine['mean']
            cov1 = combine['cov']
            eigvecs1 = combine['vecs']
            
            mat = np.concatenate((subset_X[:,0:(104*j)],subset_X[:,(104*j):(104*(j+1))]),axis = 1) - mean1
            sub_l = np.concatenate((subset_l[:,0:(104*j)],subset_l[:,(104*j):(104*(j+1))]),axis = 1) 
            W_n = (mat.T).dot(eigvecs1)
            testing_mat = testing_X-mean1
            W = (testing_mat.T).dot(eigvecs1)
            err_mat = np.full((1,104),10000)

            diff_mat = np.full((len(W_n[:,1]),M[i]),10000)
            
            for k in range(len(W[:,1])):
                diff_mat[:,:] = W[k,0:M[i]] - W_n[:,0:M[i]]
                err = np.linalg.norm(diff_mat,axis = 1)
                err_mat[0,k] = err.argmin()
            ID_real = np.reshape(testing_l,(104,))    
            ID_recog = np.reshape(sub_l[:,err_mat[0,:]].astype(int),(104,))
            right = 0        
            for n in range(len(ID_real)):
                if(ID_real[n]==ID_recog[n]):
                    right = right+1
                        
            acc_1.append(right/104)
        counter_1.append(j+1)  
    ipca_t.append(process_time_1)
    ipca_counter.append(counter_1)
    ipca_acc.append(acc_1)        
    

batch_counter = []
batch_acc = []
batch_t = []
#batch_t2 = []
time_1 = []
time_2 = []
for k in range(4):
    t_start1 = time.time()
    data = cal_data(subset_X[:,0:104*(k+1)])
    time_1.append(time.time()-t_start1)
    
    sub_l = subset_l[:,0:104*(k+1)]
    mean,A,S,eigvals,eigvecs = data['mean_data'],data['data_mat'],data['data_cov'],data['vals'],data['vecs']

    W_n = (A.T).dot(eigvecs)

    testing_mat = testing_X-mean

    W = (testing_mat.T).dot(eigvecs)

    err_mat = np.full((4,104),10000)
    N = len(W_n[1,:])#多少个eigenvector
    M = [N//8,N//4,N//2,N-1]
#    process_time = []
    for i in range(4):
#        t_start = time.time()
        diff_mat = np.full((len(W_n[:,1]),M[i]),10000)   

        for j in range(len(W[:,1])):
            diff_mat[:,:] = W[j,0:M[i]] - W_n[:,0:M[i]]
            err = np.linalg.norm(diff_mat,axis = 1)
            err_mat[i,j] = err.argmin()

    counter = []
    acc = []  
    ID_real = np.reshape(testing_l,(104,)) 
#    sub_l
    for m in range(len(err_mat[:,1])):
        ID_recog = np.reshape(sub_l[:,err_mat[m,:]].astype(int),(104,))
        right = 0        
        for n in range(len(ID_real)):
            if(ID_real[n]==ID_recog[n]):
                right = right+1
        
        acc.append(right/104)
        counter.append(m+1)
    batch_acc.append(acc)
    batch_counter.append(counter)
    batch_t.append(time_1)    
    
pca_t = batch_t[0]    
pca_clk = batch_counter[0]

ipca_M1_t = ipca_t[0] 
ipca_M2_t = ipca_t[1] 
ipca_M3_t = ipca_t[2] 
ipca_M4_t = ipca_t[3]
ipca_clk = ipca_counter[0]


err = np.full((4,104),1)
err1 = ipca_acc[0] #M = N/8 ERROR
err2 = ipca_acc[1] #M = N/4 ERROR
err3 = ipca_acc[2] #M = N/2 ERROR
err4 = ipca_acc[3] #M = N ERROR


err_matrix = np.reshape(batch_acc,(4,4))
err5 = err_matrix[:,0]
err6 = err_matrix[:,1]
err7 = err_matrix[:,2]
err8 = err_matrix[:,3]    

x1 = ipca_counter[0]
y1 = err1
y2 = err2
y3 = err3
y4 = err4
x2 = batch_counter[0]
y5 = err5
y6 = err6
y7 = err7
y8 = err8
plt.figure()

plt.figure()

plt.plot(x1,y1, color='darkgreen', marker='o',linewidth = 2,label = 'IPCA:M=N_1/8')
#plt.subplot(141),
plt.plot(x1,y2, color='green', marker='*',linewidth = 2,label = 'IPCA:M=N_1/4')
#plt.subplot(142),
plt.plot(x1,y3, color='mediumseagreen', marker='x',linewidth = 2,label = 'IPCA:M=N_1/2')
#plt.subplot(142),
plt.plot(x1,y4, color='limegreen', marker='.',linewidth = 2,label = 'IPCA:M=N_1-1')
#plt.subplot(141),
plt.plot(x2,y5, color='blue', linestyle='--', marker='o',linewidth = 2,label = 'Batch:M=N_2/8')
#plt.subplot(143),
plt.plot(x2,y6, color='royalblue', linestyle='--', marker='*',linewidth = 2,label = 'Batch:M=N_2/4')
#plt.subplot(144),
plt.plot(x2,y7, color='cornflowerblue', linestyle='--', marker='x',linewidth = 2,label = 'Batch:M=N_2/2')
#plt.subplot(144),
plt.plot(x2,y8, color='slategrey', linestyle='--', marker='.',linewidth = 2,label = 'Batch:M=N_2-1')
#plt.subplot(143),

plt.legend()
x_ticks = np.arange(0,5,1)
y_ticks = np.arange(0.25,0.64,0.05)
plt.xlim((0, 6.5))
plt.ylim((0.25, 0.64))
    #sns.regplot(x=Ms.reshape(-1, 1), y=np.array(test_dur))
plt.title('Recognition Accuracy versus Number of Training Set $\mathcal{n}$\n with Different Principle Components $\mathcal{M}$')
plt.xlabel('Number of Training Subset $\mathcal{n}$\n')
plt.ylabel('Recognition Accuracy')
#plt.legend(['IPCA:M=N/8', 'IPCA:M=N/4','IPCA:M=N/2','IPCA:M=N','PCA:M=N/8','PCA:M=N/4','PCA:M=N/2','PCA:M=N'])
plt.savefig('img/Recognition Accuracy Comparison.png',
                 dpi=1000, transparent=True)



