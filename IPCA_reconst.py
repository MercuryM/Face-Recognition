import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from scipy.linalg import *

#split data preprocessor
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
    

start_time = time.time()

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
rec_err = []
counter = []                
for i in range(4):
    data1 = cal_data(subset1_X)
    mean1,mat1,cov1,eigvals1,eigvecs1 = data1['mean_data'],data1['data_mat'],data1['data_cov'],data1['vals'],data1['vecs']
    counter = []
    for j in range(4):
        
        if j == 0:
            
            eigvecs1 = eigvecs1[:,0:M[i]]
            W_n = (mat1.T).dot(eigvecs1)
            rec_face = mean1 + eigvecs1.dot(W_n.T)
            err = np.mean(np.linalg.norm(subset1_X-rec_face,axis = 0))
            rec_err.append(err)
            counter.append(j+1)
            
        else:
            combine = combine_data(subset_X[:,0:(104*j)],subset_X[:,(104*j):(104*(j+1))], mean1, cov1, eigvecs1, M[i])
            mean1 = combine['mean']
            cov1 = combine['cov']
            eigvecs1 = combine['vecs']
            mat = np.concatenate((subset_X[:,0:(104*j)],subset_X[:,(104*j):(104*(j+1))]),axis = 1) - mean1
            W_n = (mat.T).dot(eigvecs1)
            rec_face = eigvecs1.dot(W_n.T)
            err = np.mean(np.linalg.norm(mat-rec_face, axis = 0))
            rec_err.append(err)
            counter.append(j+1)
            
            
batch_err = []
counter_1 = []
for m in range(4):
    data = cal_data(subset_X[:,0:104*(m+1)])
    mean_face,A,S,eigvals,eigvecs = data['mean_data'],data['data_mat'],data['data_cov'],data['vals'],data['vecs']

    W_n = (A.T).dot(eigvecs)
    N = len(A[1,:])
    M = [N//8,N//4,N//2,N-1]
    for n in range(4):
        rec_face =  eigvecs[:,0:M[n]].dot((W_n.T)[0:M[n],:])
        err = np.mean(np.linalg.norm(A-rec_face,axis = 0))
        batch_err.append(err)
    counter_1.append(m+1)
idx = [0,4,8,12]
err1 = []
err2 = []
err3 = []
err4 = []
for p in range(4):   #M=n/8为
    err1.append(rec_err[p]) #M = N/8 ERROR
    err2.append(rec_err[p+4]) #M = N/4 ERROR
    err3.append(rec_err[p+8]) #M = N/2 ERROR
    err4.append(rec_err[p+12]) #M = N ERROR

err5 = []
err6 = []
err7 = []
err8 = []
for q in range(4):

    err5.append(batch_err[idx[q]]) #M = N/8 ERROR
    err6.append(batch_err[idx[q]+1]) #M = N/4 ERROR
    err7.append(batch_err[idx[q]+2]) #M = N/2 ERROR
    err8.append(batch_err[idx[q]+3]) #M = N ERROR    
    

x1 = counter
y1 = err1
y2 = err2
y3 = err3
y4 = err4
x2 = counter_1
y5 = err5
y6 = err6
y7 = err7
y8 = err8


plt.figure()
#plt.subplot(141),

plt.plot(x1,y1, color='darkgreen', marker='o',linewidth = 2,label = 'IPCA:M=N_1/8')
#plt.subplot(141),
plt.plot(x1,y2, color='green', marker='*',linewidth = 2,label = 'IPCA:M=N_1/4')
#plt.subplot(142),
plt.plot(x1,y3, color='mediumseagreen', marker='x',linewidth = 2,label = 'IPCA:M=N_1/2')
#plt.subplot(142),
plt.plot(x1,y4, color='limegreen', marker='.',linewidth = 2,label = 'IPCA:M=N_1-1')
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
y_ticks = np.arange(0,1300,100)
plt.xlim((0, 6.5))
plt.ylim((-10, 1300))
    #sns.regplot(x=Ms.reshape(-1, 1), y=np.array(test_dur))
plt.title('Reconstruction Error versus Number of Training Set $\mathcal{n}$\n with Different Principle Components $\mathcal{M}$')
plt.xlabel('Number of Training Subset $\mathcal{n}$\n')
plt.ylabel('Reconstruction Error')
#plt.legend(['IPCA:M=N/8', 'IPCA:M=N/4','IPCA:M=N/2','IPCA:M=N','PCA:M=N/8','PCA:M=N/4','PCA:M=N/2','PCA:M=N'])
plt.savefig('img/Reconstruction Error Comparison.png',
                 dpi=1000, transparent=True)
