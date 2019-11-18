# Random subspace
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

from sklearn import model_selection
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from Split import split_data
from sklearn.metrics import confusion_matrix
from Visualize import plot_confusion_matrix
from numpy import *

data = split_data()
X_train, y_train = data['train']
D, N = X_train.shape
X_test, y_test = data['test']
I, K = X_test.shape
acc_sum = [array([90.38461538]), array([94.23076923]), array([92.30769231]), array([95.19230769]), array([95.19230769]), array([95.19230769]), array([94.23076923]), array([96.15384615]), array([96.15384615]), array([97.11538462]), array([96.15384615]), array([96.15384615]), array([95.19230769]), array([96.15384615]), array([96.15384615]), array([98.07692308]), array([97.11538462]), array([96.15384615]), array([95.19230769]), array([95.19230769]), array([95.19230769]), array([95.19230769]), array([97.11538462]), array([95.19230769]), array([96.15384615]), array([97.11538462]), array([96.15384615]), array([97.11538462]), array([95.19230769]), array([97.11538462]), array([97.11538462]), array([97.11538462]), array([98.07692308]), array([98.07692308]), array([98.07692308]), array([95.19230769]), array([98.07692308]), array([97.11538462]), array([96.15384615]), array([95.19230769]), array([95.19230769]), array([97.11538462]), array([96.15384615]), array([96.15384615]), array([97.11538462]), array([96.15384615]), array([96.15384615]), array([95.19230769]), array([97.11538462]), array([95.19230769]), array([96.15384615]), array([97.11538462]), array([97.11538462]), array([97.11538462]), array([96.15384615]), array([96.15384615]), array([97.11538462]), array([96.15384615]), array([98.07692308]), array([96.15384615])]


def random_subspace(n_estimators, M0, M1, verbose=False):
    if n_estimators == 0:
        ave_sub_model_acc = 0
        acc = 0
    else:
        standard = False
        # M__pca_ideal = 147
        # M__lda_ideal = 46
        # if verbose:
        #    print ('M__pca_ideal = ', M__pca_ideal)
        #    print ('M__lda_ideal = ', M__lda_ideal)

        M_pca_bag = N - 1

        M_pca = 147  # M__pca_ideal
        M_lda = 46  # M__lda_ideal

        assert (M1 <= (N - 1 - M0))
        assert (M0 + M1 > M_lda)

        estimators = [('lda', LinearDiscriminantAnalysis(n_components=M_lda)), ('knn', KNeighborsClassifier(n_neighbors=1))]

        base_est = Pipeline(estimators)

        base_est.fit(X_train.T, y_train.T.ravel())

        acc = base_est.score(X_test.T, y_test.T.ravel())
        if verbose:
            print('Accuracy of base estimator with no pre PCA = %.2f%%' % (acc * 100))

        pca = PCA(n_components=M_pca_bag)
        W_train = pca.fit_transform(X_train.T)
        W_test = pca.transform(X_test.T)

        base_est.fit(W_train, y_train.T.ravel())

        acc = base_est.score(W_test, y_test.T.ravel())
        if verbose:
            print('Accuracy of base estimator with pre PCA applied = %.2f%%' % (acc * 100))

        estimators = []
        sub_model_accuracies = []
        masks = []

        for i in range(n_estimators):

            mask0 = np.arange(M0)
            mask1 = np.random.choice(np.arange(M0, (N - 1)), M1, replace=False)

            mask1 = np.array(mask1).ravel()

            mask = np.concatenate((mask0, mask1), axis=None)
            masks.append(mask)

            W_bag = W_train[:, mask]
            y_bag = y_train

            estimator = clone(base_est)

            estimator.fit(W_bag, y_bag.T.ravel())

            name = 'est_' + str(i + 1)
            estimators.append((name, estimator))

            sub_model_acc = estimator.score(W_test[:, mask], y_test.T.ravel())
            sub_model_accuracies.append(sub_model_acc)
            if verbose:
                print('Accuracy of sub model ', i + 1, ' = %.2f%%' % (sub_model_acc * 100))

        ave_sub_model_acc = sum(sub_model_accuracies) / n_estimators
        if verbose:
            print('Average accuracy of sub models = %.2f%%' % (ave_sub_model_acc * 100))

        y_hat = []

        for w in W_test:
            prediction_sum = 0
            predictions = np.empty(n_estimators, dtype=np.int64)
            for i, (name, estimator) in enumerate(estimators):
                y = estimator.predict(w[masks[i]].reshape(1, -1))

                prediction_sum = prediction_sum + float(y[0])
                predictions[i] = int(y[0])
            #sum
            prediction = round(prediction_sum / n_estimators)
            # y_hat.append(prediction)
            #voting
            counts = np.bincount(predictions)
            y_hat.append(np.argmax(counts))

        acc = accuracy_score(y_test.T, y_hat)
        if verbose:
            print('Accuracy of ensemble models = %.2f%%' % (acc * 100))

    return acc, ave_sub_model_acc, sub_model_accuracies


# n_estimators = 35
# M0 = 50
# M1 = 50
# acc, ave_sub_model_acc, sub_model_accuracies = random_subspace(n_estimators, M0, M1, verbose=True)
# print(1-acc)
# print(1-ave_sub_model_acc)


# x = np.arange(1, 36, 1)
# acc_voting = acc*100 + x - x
# acc_sum = 98.07692308 + x - x
# plt.figure(figsize=(8.0, 6.0))
# plt.plot(x, acc_sum, marker = 'x', label='Combining 35 models using sum rule')
# plt.plot(x, acc_voting, color='mediumseagreen', marker = '|', label='Combine 35 models using majority voting rule')
# plt.plot(x, sub_model_accuracies, marker = 'o', label='Individual Model')
# plt.title('Accuracy versus n_estimators\n')
# plt.xlabel('n_estimators')
# plt.ylabel('Recogniton Accuracy of 35 models[%]')
#
# plt.legend(loc='best')
# plt.savefig('img/accuracy_versus_35.png',
#                 dpi=1000, transparent=True)
# plt.show()


D, N = X_train.shape

n_estimators = 30
M0 = 0
M1 = 147 - M0 + 1
#
acc_varying_subspace = []
num_M0 = []
num_M1 = []
#
M0_ideal = None
M1_ideal = None
acc_max = 0
#
while M0 <= N - 1:
   M1 = max((47 - M0 + 1), 0)
   num_M1_i = []
   acc_varying_subspace_i = []
   while M1 <= (N - 1 - M0):
       acc, ave_sub_model_acc, sub_model_accuracies = random_subspace(n_estimators, M0, M1)
       acc_varying_subspace_i.append((acc * 100))
       num_M1_i.append(M1)

       if (acc > acc_max):
           M0_ideal = M0
           M1_ideal = M1
           acc_max = acc

       M1 = M1 + 20

   num_M1.append(num_M1_i)
   acc_varying_subspace.append(acc_varying_subspace_i)
   num_M0.append(M0)
   M0 = M0 + 50

print("Accuracy is maximum for M0 = ", M0_ideal, ", M1 = ", M1_ideal, " with accuracy of %.2f%%" % (acc_max * 100), ".")

# Accuracy is maximum for M0 =  150 , M1 =  80  with accuracy of 97.12% .

# n_estimators = 1
# M0 = 150
# M1 = 80
# n_est_test_range = 60
# #
# num_estimators_list = []
# acc_varying_num_est_ran_subsp = []
# acc_sub_model_ave = []
# #
# while n_estimators <= n_est_test_range:
#    acc, ave_sub_model_acc = random_subspace(n_estimators, M0, M1)
#    acc_varying_num_est_ran_subsp.append(acc * 100)
#    acc_sub_model_ave.append(ave_sub_model_acc * 100)
#    num_estimators_list.append(n_estimators)
#    n_estimators = n_estimators + 1
#
# acc_sum = [array([90.38461538]), array([94.23076923]), array([92.30769231]), array([95.19230769]), array([95.19230769]), array([95.19230769]), array([94.23076923]), array([96.15384615]), array([96.15384615]), array([97.11538462]), array([96.15384615]), array([96.15384615]), array([95.19230769]), array([96.15384615]), array([96.15384615]), array([98.07692308]), array([97.11538462]), array([96.15384615]), array([95.19230769]), array([95.19230769]), array([95.19230769]), array([95.19230769]), array([97.11538462]), array([95.19230769]), array([96.15384615]), array([97.11538462]), array([96.15384615]), array([97.11538462]), array([95.19230769]), array([97.11538462]), array([97.11538462]), array([97.11538462]), array([98.07692308]), array([98.07692308]), array([98.07692308]), array([95.19230769]), array([98.07692308]), array([97.11538462]), array([96.15384615]), array([95.19230769]), array([95.19230769]), array([97.11538462]), array([96.15384615]), array([96.15384615]), array([97.11538462]), array([96.15384615]), array([96.15384615]), array([95.19230769]), array([97.11538462]), array([95.19230769]), array([96.15384615]), array([97.11538462]), array([97.11538462]), array([97.11538462]), array([96.15384615]), array([96.15384615]), array([97.11538462]), array([96.15384615]), array([98.07692308]), array([96.15384615])]
# plt.figure(figsize=(8.0, 6.0))
# plt.plot(num_estimators_list, acc_sum, label='Accuracy of Ensemble Model using sum rule')
# plt.plot(num_estimators_list, acc_varying_num_est_ran_subsp, color='mediumseagreen', label='Accuracy of Ensemble Model using majority voting rule')
# plt.plot(num_estimators_list, acc_sub_model_ave, label='Average Accuracy of Individual Models')
# plt.title('Accuracy versus n_estimators\n')
# plt.xlabel('n_estimators')
# plt.ylabel('Recogniton Accuracy [%]')
# plt.legend(loc='best')
# plt.savefig('img/accuracy_versus_n_estimators_subspace.png',
#                 dpi=1000, transparent=True)
# plt.show()


print (num_M1[3])
plt.figure(figsize=(8.0, 6.0))
color_list = ['green', 'blue', 'red', 'purple', 'orange', 'magenta', 'cyan', 'black', 'indianred', 'lightseagreen', 'gold']
for i in range(len(num_M0)):
   plt.plot(num_M1[i], acc_varying_subspace[i], color=color_list[i], linestyle='dashed', label='$\mathcal{M0}$ = '+str(num_M0[i]))
plt.title('Recogniton Accuracy [%] versus nuber of Random Dimensions ($\mathcal{M1}$) for a range of fixed $\mathcal{M0}$')
plt.xlabel('Number of Randomly Selected Dimensions $\mathcal{M1}$')
plt.ylabel('Recogniton Accuracy [%]')
plt.legend(loc='best')
plt.savefig('img/Accuracy_versus_M1_for_a_range_of_fixed_M0.png', dpi=1000, Transparent = True)


# plt.figure(figsize=(8.0, 6.0))
# plt.plot(num_estimators_list, acc_varying_num_est_bag, color='green', linestyle='dashed', label='Bagging')
# plt.plot(num_estimators_list, acc_varying_num_est_ran_subsp, color='blue', linestyle='dashed', label='Random Subspace')
# plt.title('Accuracy vs number of estimators\n')
# plt.xlabel('Number of estimators')
# plt.ylabel('Recogniton Accuracy / %')
# plt.legend(loc='best')