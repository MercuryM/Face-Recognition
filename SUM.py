#import numpy as np
#import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split
#from sklearn.neighbors import KNeighborsClassifier
#import scipy.io as sio
#from scipy.stats import mode
#from sklearn.metrics.pairwise import cosine_similarity
#def data_split(pixels, labels, test_ratio):
#     np.random.seed(13)
#     test_im = np.zeros((1,2576))
#     test_l = np.zeros((1,1))
#     choice_all = np.zeros((1,))
#     train_im = np.copy(pixels)
#     train_l = np.copy(labels)
#     for i in range(52):
#         choice = np.random.choice(10, test_ratio, replace = False)+i*10
#         #print(choice.shape)
#         choice_all = np.append(choice_all, choice, axis = 0)
#         test_im = np.append(test_im, pixels[choice], axis = 0)
#         test_l = np.append(test_l, labels[choice], axis = 0)
#     choice_all = np.delete(choice_all, 0, 0).astype(np.int16)
#     train_im = np.delete(train_im, choice_all, 0)
#     train_l = np.delete(train_l, choice_all, 0)
#     test_im = np.delete(test_im, 0, 0)
#     test_l = np.delete(test_l, 0, 0)
#     return train_im, test_im, train_l, test_l
#
#def feature_sampling_model(N, num_model, M_0, M_1):
#     feature_choice = np.zeros((num_model, M_0 + M_1))
#     for i in range(num_model):
#         choice = np.random.choice(N - M_0, M_1, replace = False) + M_0
#         feature_choice[i] = np.hstack((np.arange(M_0), choice))
#     return feature_choice
#def normal_computation(train_image):
#     cov_mat = np.cov(train_image.T)
#     #print('Covariance matrix: \n%s' %cov_mat)
#     return np.linalg.eigh(cov_mat)
#def mean_face(train_image):
#     #train_im.T is D*N
#     #x_mean is D*1
#     x_mean = np.mean(train_image.T, axis = 1).reshape(len(train_image.T),1)
#     return x_mean
#def face_classification(train_image, train_label, test_image, eigenface):
#     train_w = eigenface.T@train_image #M*N
#     knn = KNeighborsClassifier(n_neighbors=1)
#     knn.fit(train_w.T, train_label.ravel())
#     # print('d')
#     # print((train_label.tolist()).shape)
#     test_matrix = eigenface.T@test_image #M*T
#     classified_label = knn.predict(test_matrix.T)
#     classified_label = classified_label.reshape(len(classified_label),1)
#     return classified_label
#
#def sum_rule(W, l_mean, test_image):
#     means_project = W.T@l_mean
#     data_project = W.T@test_image.T
#     norm_x = np.linalg.norm(data_project, axis = 0).reshape(1,len(np.linalg.norm(data_project, axis = 0))) #1*n
#     norm_means = np.linalg.norm(means_project, axis = 0).reshape(1,len(np.linalg.norm(means_project, axis = 0))) #1*52
#     p = (1 + (data_project.T@means_project)/(norm_x.T@norm_means))/2
#     return p
#
#split_number = 2
#height, width = (46, 56)
#D = height * width
#architecture_flag = 1 # 0: Data Sampling 1: FeatureSpace Sampling
#T = 30
#faces = sio.loadmat('face.mat')
#face_pixels = faces['X'].T
#face_labels = faces['l'].T
#train_im, test_im, train_l, test_l = data_split(face_pixels,face_labels, split_number)
#print(train_l)
#
#if architecture_flag == 1:
#     M0 = 0
#     M1 = 147 - M0 + 1
#     M0_ideal = None
#     M1_ideal = None
#     acc_max = 0
#     S_w = np.zeros((D, D))
#     S_b = np.zeros((D, D))
#     N_nonzero = len(train_im) - 1
#     eigen_vals, eigen_vecs = normal_computation(train_im)
#     eigen_vals_sorted = eigen_vals[(-1 * (np.abs(eigen_vals))).argsort()]
#     eigen_faces = eigen_vecs.T[((-1 * (np.abs(eigen_vals))).argsort())]
#     eigen_faces = eigen_faces.T[:, 0:len(train_im) - 1]
#     # #
#     while M0 <= 416 - 1:
#        M1 = max((147 - M0 + 1), 0)
#        while M1 <= (416 - 1 - M0):
#            M_lda = 46
#            choice = (feature_sampling_model(N_nonzero, T, M0, M1)).astype(np.int16)
#            classified_label = np.zeros((T, len(test_l)))
#            classified_p = np.zeros((len(test_l), 52))
#            local_means = np.zeros((52, D))
#            for i in range(52):
#                local_means[i] = (mean_face(train_im[8*i:8*(i+1)]).T)
#            local_means = local_means.T
#            S_b = np.cov(local_means)
#            for j in range(52):
#                class_Sw = np.cov(train_im[8*j:8*(j+1),:].T)
#                S_w = S_w + class_Sw
#            for j in range(T):
#                W_pca = eigen_faces.T[choice[j]].T
#                fisher_vals, fisher_faces = np.linalg.eig(np.linalg.inv(W_pca.T@S_w@W_pca)@(W_pca.T@S_b@W_pca))
#                fisher_faces = fisher_faces.T[((-1*(np.abs(fisher_vals))).argsort())]
#                projections_pca_train = W_pca.T@train_im.T
#                projections_pca_test = W_pca.T@test_im.T
#                fisher_faces = fisher_faces.T
#                W_lda = fisher_faces[:,0:M_lda]
#                W_opt = W_pca@W_lda
#                # classified_label = face_classification(projections_pca_train, train_l, projections_pca_test, W_opt, M_lda) classified_label[j] = (face_classification(train_im.T, train_l, test_im.T, W_opt)).T
#                classified_p = classified_p + sum_rule(W_opt, local_means, test_im)
#                classified_l = np.argmax(classified_p, axis = 1)+1
#                classified_l = classified_l.reshape(1,len(classified_l))
#                test_l = test_l.astype(np.int16)
#                classified_table = (classified_l.T == test_l)
#                classified_table_num = classified_table.astype(np.int16)
#                acc = (sum(classified_table_num)/len(classified_table_num))
#                classified_table = classified_table_num.reshape((52, split_number))
#                print(classified_table)
#                # print(pd.DataFrame(classified_table,columns = ['Sample 1','Sample 2','Sample 3']))
#                print('\nCorrectly Classified Percentage:')
#                print((sum(classified_table_num)/len(classified_table_num)))
#                if (acc > acc_max):
#                       M0_ideal = M0
#                       M1_ideal = M1
#                       acc_max = acc
#            M1 = M1 + 20
#        M0 = M0 + 50
#     print("Accuracy is maximum for M0 = ", M0_ideal, ", M1 = ", M1_ideal, " with accuracy of %.2f%%" % (acc_max * 100), ".")
#
# else:
#     M_pca = 150 #312
#     M_lda = 40 #51
#     N_t = 7
#     #choice = data_sampling_model(train_im, T, N_t)
#     classified_label = np.zeros((T, len(test_l)))
#     classified_p = np.zeros((len(test_l), 52))
#     for k in range(T):
#         S_w = np.zeros((D,D))
#         S_b = np.zeros((D,D))
#         local_means = np.zeros((52, D))
#         sample_im = train_im
#         sample_l = train_l
#         eigen_vals, eigen_vecs = normal_computation(sample_im)
#         eigen_vals_sorted = eigen_vals[(-1*(np.abs(eigen_vals))).argsort()]
#         print('Model ' + str(k) + ' in training...')
#         eigen_faces = eigen_vecs.T[((-1*(np.abs(eigen_vals))).argsort())]
#         eigen_faces = eigen_faces.T
#         W_pca = eigen_faces[:,0:M_pca]
#         global_mean = mean_face(sample_im)
#         for i in range(52):
#             local_means[i] = (mean_face(sample_im[N_t*i:N_t*(i+1)]).T)
#             #S_B += (local_means[i].T-global_mean)@(local_means[i].T-global_mean).
#             local_means = local_means.T
#             #sb = (local_means - np.tile(global_mean,(1,len(local_means.T))))@(local_means -np.tile(global_mean,(1,len(local_means.T)))).T
#         S_b = np.cov(local_means)
#         for j in range(52):
#             class_Sw = np.cov(sample_im[N_t*j:N_t*(j+1),:].T)
#             S_w = S_w + class_Sw
#          #print(S_w)
#         fisher_vals, fisher_faces = np.linalg.eig(np.linalg.inv(W_pca.T @ S_w @ W_pca) @ (W_pca.T @ S_b @ W_pca))
#         fisher_faces = fisher_faces.T[((-1 * (np.abs(fisher_vals))).argsort())]
#         projections_pca_train = W_pca.T @ sample_im.T
#         projections_pca_test = W_pca.T @ sample_im.T
#         fisher_faces = fisher_faces.T
#         W_lda = fisher_faces[:, 0:M_lda]
#         W_opt = W_pca @ W_lda
#         # classified_label = face_classification(projections_pca_train, train_l, projections_pca_test, W_opt, M_lda)
#         classified_label[k] = (face_classification(sample_im.T, sample_l, test_im.T, W_opt)).T
#         classified_p = classified_p + sum_rule(W_opt, local_means, test_im)
#         classified_l = np.argmax(classified_p, axis=1) + 1
#         classified_l = classified_l.reshape(1, len(classified_l))
#         test_l = test_l.astype(np.int16)
#         classified_table = (classified_l.T == test_l)
#         classified_table_num = classified_table.astype(np.int16)
#         classified_table = classified_table_num.reshape((52, split_number))
# classified_l, count = mode(classified_label, axis=0)
# classified_l = np.argmax(classified_p, axis = 1)+1
# classified_l = classified_l.reshape(1,len(classified_l))
# test_l = test_l.astype(np.int16)
# classified_table = (classified_l.T == test_l)
# classified_table_num = classified_table.astype(np.int16)
# classified_table = classified_table_num.reshape((52, split_number))
# print(classified_table)
# # print(pd.DataFrame(classified_table,columns = ['Sample 1','Sample 2','Sample 3']))
# print('\nCorrectly Classified Percentage:')
# print((sum(classified_table_num)/len(classified_table_num)))
        # confusion_matrix = np.load(
        #     'coMatrix3.npy')  # confusion_matrix = np.zeros(53*53).reshape((53,53)) c_matrix = np.zeros(53*53).reshape((53,53))
        # for i in range(len(test_l)):
        #     c_matrix[test_l[i], classified_l.T[i]] += 1
        #     confusion_matrix = confusion_matrix + c_matrix
        #     np.save('coMatrix3.npy', confusion_matrix)













import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import scipy.io as sio
from scipy.stats import mode
from sklearn.metrics.pairwise import cosine_similarity
from Split import split_data
from sklearn.metrics import confusion_matrix

from Visualize import plot_confusion_matrix


def data_split(pixels, labels, test_ratio):
    train_im = np.zeros((1,2576))
    train_l = np.zeros((1,1))
    choice_all = np.zeros((1,))
    test_im = np.copy(pixels)
    test_l = np.copy(labels)
    for i in range(52):
        np.random.seed(13)
        choice = np.random.choice(10, test_ratio, replace = False)+i*10
        #print(choice.shape)
        choice_all = np.append(choice_all, choice, axis = 0)
        train_im = np.append(train_im, pixels[choice], axis = 0)
        train_l = np.append(train_l, labels[choice], axis = 0)
    choice_all = np.delete(choice_all, 0, 0).astype(np.int16)
    test_im = np.delete(test_im, choice_all, 0)
    test_l = np.delete(test_l, choice_all, 0)
    train_im = np.delete(train_im, 0, 0)
    train_l = np.delete(train_l, 0, 0)
    return train_im, test_im, train_l, test_l
def feature_sampling_model(N, num_model, M_0, M_1):
    feature_choice = np.zeros((num_model, M_0 + M_1))
    for i in range(num_model):
        choice = np.random.choice(N - M_0, M_1, replace = False) + M_0
        feature_choice[i] = np.hstack((np.arange(M_0), choice))
    return feature_choice

def normal_computation(train_image):
    cov_mat = np.cov(train_image.T)
    #print('Covariance matrix: \n%s' %cov_mat)
    return np.linalg.eigh(cov_mat)

def mean_face(train_image):
     #train_im.T is D*N
     #x_mean is D*1
     x_mean = np.mean(train_image.T, axis = 1).reshape(len(train_image.T),1)
     return x_mean
def face_classification(train_image, train_label, test_image, eigenface):
    train_w = eigenface.T@train_image #M*N
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(train_w.T, train_label.ravel())
    # print('d')
    # print((train_label.tolist()).shape)
    test_matrix = eigenface.T@test_image #M*T
    classified_label = knn.predict(test_matrix.T)
    classified_label = classified_label.reshape(len(classified_label),1)
    return classified_label

def sum_rule(W, l_mean, test_image):
    means_project = W.T@l_mean
    data_project = W.T@test_image.T
    norm_x = np.linalg.norm(data_project, axis = 0).reshape(1,len(np.linalg.norm(data_project, axis = 0))) #1*n
    norm_means = np.linalg.norm(means_project, axis = 0).reshape(1,len(np.linalg.norm(means_project, axis = 0))) #1*52
    p = (1 + (data_project.T@means_project)/(norm_x.T@norm_means))/2
    return p

data = split_data()
train_im, train_l = data['train']
test_im, test_l = data['test']
train_im = train_im.T
test_im = test_im.T
train_l = train_l.T
test_l = test_l.T

height, width = (46, 56)
D = height * width
#architecture_flag = 1 # 0: Data Sampling 1: FeatureSpace Sampling
T = 60

# def rad_sum_classifier (T, train_im, train_l, test_im, test_l, D):
Num_sample, M0, M1 = 416, 150, 80
M_lda = 46
S_w = np.zeros((D, D))
S_b = np.zeros((D, D))
N_nonzero = len(train_im) - 1
eigen_vals, eigen_vecs = normal_computation(train_im)
eigen_vals_sorted = eigen_vals[(-1 * (np.abs(eigen_vals))).argsort()]
eigen_faces = eigen_vecs.T[((-1 * (np.abs(eigen_vals))).argsort())]
eigen_faces = eigen_faces.T[:, 0:len(train_im) - 1]
    # #
choice = (feature_sampling_model(N_nonzero, T, M0, M1)).astype(np.int16)
classified_label = np.zeros((T, len(test_l)))
classified_p = np.zeros((len(test_l), 52))
local_means = np.zeros((52, D))
for i in range(52):
    local_means[i] = (mean_face(train_im[8*i:8*(i+1)]).T)
local_means = local_means.T
S_b = np.cov(local_means)
for j in range(52):
    class_Sw = np.cov(train_im[8*j:8*(j+1),:].T)
    S_w = S_w + class_Sw
for j in range(T):
    W_pca = eigen_faces.T[choice[j]].T
    fisher_vals, fisher_faces = np.linalg.eig(np.linalg.inv(W_pca.T@S_w@W_pca)@(W_pca.T@S_b@W_pca))
    fisher_faces = fisher_faces.T[((-1*(np.abs(fisher_vals))).argsort())]
    projections_pca_train = W_pca.T@train_im.T
    projections_pca_test = W_pca.T@test_im.T
    fisher_faces = fisher_faces.T
    W_lda = fisher_faces[:,0:M_lda]
    W_opt = W_pca@W_lda
    # classified_label = face_classification(projections_pca_train, train_l, projections_pca_test, W_opt, M_lda) classified_label[j] = (face_classification(train_im.T, train_l, test_im.T, W_opt)).T
    classified_p = classified_p + sum_rule(W_opt, local_means, test_im)
classified_l = np.argmax(classified_p, axis = 1)+1
classified_l = classified_l.reshape(1,len(classified_l))
test_l = test_l.astype(np.int16)
classified_table = (classified_l.T == test_l)
classified_table_num = classified_table.astype(np.int16)
acc = (sum(classified_table_num)/len(classified_table_num))
classified_table = classified_table_num.reshape((52, 2))
print(classified_table)
    # print(pd.DataFrame(classified_table,columns = ['Sample 1','Sample 2','Sample 3']))
print('\nCorrectly Classified Percentage:')
print((sum(classified_table_num)/len(classified_table_num)))
# return acc


#
# data = split_data()
# train_im, train_l = data['train']
# test_im, test_l = data['test']
# train_im = train_im.T
# test_im = test_im.T
# train_l = train_l.T
# test_l = test_l.T
#
# height, width = (46, 56)
# D = height * width


# n_estimators = 1
# n_est_test_range = 60
# num_estimators_list = []
# acc_varying_num_est_ran_subsp = []
# acc_sub_model_ave = []
# #
# while n_estimators <= n_est_test_range:
#    acc =  rad_sum_classifier (n_estimators, train_im, train_l, test_im, test_l, D)
#    acc_varying_num_est_ran_subsp.append(acc * 100)
#    num_estimators_list.append(n_estimators)
#    n_estimators = n_estimators + 1
#
# plt.figure(figsize=(8.0, 6.0))
# plt.plot(num_estimators_list, acc_varying_num_est_ran_subsp)
# plt.title('Accuracy versus n_estimators\n')
# plt.xlabel('n_estimators')
# plt.ylabel('Recogniton Accuracy [%]')
# plt.legend(loc='best')
# plt.savefig('img/sum_accuracy_versus_n_estimators_subspace.png',
#                 dpi=1000, transparent=True)
# plt.show()
#else:
#     M_pca = 150 #312
#     M_lda = 40 #51
#     N_t = 7
#     #choice = data_sampling_model(train_im, T, N_t)
#     classified_label = np.zeros((T, len(test_l)))
#     classified_p = np.zeros((len(test_l), 52))
#     for k in range(T):
#         S_w = np.zeros((D,D))
#         S_b = np.zeros((D,D))
#         local_means = np.zeros((52, D))
#         sample_im = train_im
#         sample_l = train_l
#         eigen_vals, eigen_vecs = normal_computation(sample_im)
#         eigen_vals_sorted = eigen_vals[(-1*(np.abs(eigen_vals))).argsort()]
#         print('Model ' + str(k) + ' in training...')
#         eigen_faces = eigen_vecs.T[((-1*(np.abs(eigen_vals))).argsort())]
#         eigen_faces = eigen_faces.T
#         W_pca = eigen_faces[:,0:M_pca]
#         global_mean = mean_face(sample_im)
#         for i in range(52):
#             local_means[i] = (mean_face(sample_im[N_t*i:N_t*(i+1)]).T)
#             #S_B += (local_means[i].T-global_mean)@(local_means[i].T-global_mean).
#             local_means = local_means.T
#             #sb = (local_means - np.tile(global_mean,(1,len(local_means.T))))@(local_means -np.tile(global_mean,(1,len(local_means.T)))).T
#         S_b = np.cov(local_means)
#         for j in range(52):
#             class_Sw = np.cov(sample_im[N_t*j:N_t*(j+1),:].T)
#             S_w = S_w + class_Sw
#          #print(S_w)
#         fisher_vals, fisher_faces = np.linalg.eig(np.linalg.inv(W_pca.T @ S_w @ W_pca) @ (W_pca.T @ S_b @ W_pca))
#         fisher_faces = fisher_faces.T[((-1 * (np.abs(fisher_vals))).argsort())]
#         projections_pca_train = W_pca.T @ sample_im.T
#         projections_pca_test = W_pca.T @ sample_im.T
#         fisher_faces = fisher_faces.T
#         W_lda = fisher_faces[:, 0:M_lda]
#         W_opt = W_pca @ W_lda
#         # classified_label = face_classification(projections_pca_train, train_l, projections_pca_test, W_opt, M_lda)
#         classified_label[k] = (face_classification(sample_im.T, sample_l, test_im.T, W_opt)).T
#         classified_p = classified_p + sum_rule(W_opt, local_means, test_im)
#         classified_l = np.argmax(classified_p, axis=1) + 1
#         classified_l = classified_l.reshape(1, len(classified_l))
#         test_l = test_l.astype(np.int16)
#         classified_table = (classified_l.T == test_l)
#         classified_table_num = classified_table.astype(np.int16)
#         classified_table = classified_table_num.reshape((52, 10 - split_number))


data = split_data()
X_train, y_train = data['train']
X_test, y_test = data['test']

classes = set(y_train.ravel())
print(y_test.T.shape)
print(classified_l.T.shape)
cnf_matrix = confusion_matrix(y_test.T, classified_l.T, labels=list(classes))
plt.rcParams['figure.figsize'] = [28.0, 21.0]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True,
                     title='Sum rule - Normalized Confusion Matrix',
                     cmap=plt.cm.Blues)

plt.savefig('img/sum_nn_cnf_matrix.png', dpi=300, transparent=True)



#import numpy as np
#import matplotlib.pyplot as plt
#from sklearn.ensemble import BaggingClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.pipeline import Pipeline
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.decomposition import PCA
#
#from sklearn import model_selection
#from sklearn.ensemble import VotingClassifier
#from sklearn.metrics import accuracy_score
#from sklearn.base import clone
#from Split import split_data
#from sklearn.metrics import confusion_matrix
#from Visualize import plot_confusion_matrix
#
#data = split_data()
#X_train, y_train = data['train']
#D, N = X_train.shape
#X_test, y_test = data['test']
#I, K = X_test.shape
#
#
#standard = False
#M__pca_ideal = 147
#M__lda_ideal = 46
#
#print('M__pca_ideal = ', M__pca_ideal)
#print('M__lda_ideal = ', M__lda_ideal)
#
#M_pca_bag = N - 1
#
#M_pca = 150  # M__pca_ideal
#M_lda = 47  # M__lda_ideal
#
#n_est = 10
#
#estimators = [('pca', PCA(n_components=M_pca)), ('lda', LinearDiscriminantAnalysis(n_components=M_lda)),
#              ('knn', KNeighborsClassifier(n_neighbors=1))]
#
#base_est = Pipeline(estimators)
#
#print(X_train.shape)
#print(y_train.shape)
#
#base_est.fit(X_train.T, y_train.T.ravel())
#
#acc = base_est.score(X_test.T, y_test.T.ravel())
#print('Accuracy of base estimator with no pre PCA = %.2f%%' % (acc * 100))
#
#pca = PCA(n_components=M_pca_bag)
#W_train = pca.fit_transform(X_train.T)
#W_test = pca.transform(X_test.T)
#
#base_est.fit(W_train, y_train.T.ravel())
#
#acc = base_est.score(W_test, y_test.T.ravel())
#print('Accuracy of base estimator with pre PCA applied = %.2f%%' % (acc * 100))
#
#bagging = BaggingClassifier(base_estimator=base_est,
#                            max_samples=1.0,
#                            max_features=1.0,
#                            bootstrap=True,
#                            # bootstrap_features=True,
#                            n_estimators=n_est)
#
#print(W_train.shape)
#
#bagging = bagging.fit(W_train, y_train.T.ravel())
#
#acc = bagging.score(W_test, y_test.T.ravel())
#print('Accuracy of ensemble estimator = %.2f%%' % (acc * 100))
#
#seed = 7
#kfold = model_selection.KFold(n_splits=10, random_state=seed)
#results = model_selection.cross_val_score(bagging, W_train, y_train.T.ravel(), cv=kfold)
#print('Cross validation mean accuracy = %.2f%%' % (results.mean() * 100))
#
#sub_model_accuracies = []
#
#sub_estimators = []
#
#for i, estimator in enumerate(bagging.estimators_):
#    print(estimator)
#    sub_model_acc = estimator.score(W_test, y_test.T.ravel())
#    print('Accuracy of sub model ', i + 1, ' = %.2f%%' % (sub_model_acc * 100))
#    sub_model_accuracies.append(sub_model_acc)
#    name = 'est' + str(i + 1)
#    sub_estimators.append((name, estimator))
#
#ave_sub_model_acc = sum(sub_model_accuracies) / n_est
#
#print('Average accuracy of sub models = %.2f%%' % (ave_sub_model_acc * 100))
#
#voting = VotingClassifier(estimators=sub_estimators, voting='soft')
#voting = voting.fit(W_train, y_train.T.ravel())
#acc = voting.score(W_test, y_test.T.ravel())
#print('Accuracy of voting = %.2f%%' % (acc * 100))



