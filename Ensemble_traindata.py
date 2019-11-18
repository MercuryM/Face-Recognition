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

data = split_data()
X_train, y_train = data['train']
D, N = X_train.shape
X_test, y_test = data['test']
I, K = X_test.shape


# standard = False
# M__pca_ideal = 147
# M__lda_ideal = 46
#
# print('M__pca_ideal = ', M__pca_ideal)
# print('M__lda_ideal = ', M__lda_ideal)
#
# M_pca_bag = N - 1
#
# M_pca = 150  # M__pca_ideal
# M_lda = 47  # M__lda_ideal
#
# n_est = 10
#
# estimators = [('pca', PCA(n_components=M_pca)), ('lda', LinearDiscriminantAnalysis(n_components=M_lda)),
#               ('knn', KNeighborsClassifier(n_neighbors=1))]
#
# base_est = Pipeline(estimators)
#
# print(X_train.shape)
# print(y_train.shape)
#
# base_est.fit(X_train.T, y_train.T.ravel())
#
# acc = base_est.score(X_test.T, y_test.T.ravel())
# print('Accuracy of base estimator with no pre PCA = %.2f%%' % (acc * 100))
#
# pca = PCA(n_components=M_pca_bag)
# W_train = pca.fit_transform(X_train.T)
# W_test = pca.transform(X_test.T)
#
# base_est.fit(W_train, y_train.T.ravel())
#
# acc = base_est.score(W_test, y_test.T.ravel())
# print('Accuracy of base estimator with pre PCA applied = %.2f%%' % (acc * 100))
#
# bagging = BaggingClassifier(base_estimator=base_est,
#                             max_samples=1.0,
#                             max_features=1.0,
#                             bootstrap=True,
#                             # bootstrap_features=True,
#                             n_estimators=n_est)
#
# print(W_train.shape)
#
# bagging = bagging.fit(W_train, y_train.T.ravel())
#
# acc = bagging.score(W_test, y_test.T.ravel())
# print('Accuracy of ensemble estimator = %.2f%%' % (acc * 100))
#
# seed = 7
# kfold = model_selection.KFold(n_splits=10, random_state=seed)
# results = model_selection.cross_val_score(bagging, W_train, y_train.T.ravel(), cv=kfold)
# print('Cross validation mean accuracy = %.2f%%' % (results.mean() * 100))
#
# sub_model_accuracies = []
#
# sub_estimators = []
#
# for i, estimator in enumerate(bagging.estimators_):
#     print(estimator)
#     sub_model_acc = estimator.score(W_test, y_test.T.ravel())
#     print('Accuracy of sub model ', i + 1, ' = %.2f%%' % (sub_model_acc * 100))
#     sub_model_accuracies.append(sub_model_acc)
#     name = 'est' + str(i + 1)
#     sub_estimators.append((name, estimator))
#
# ave_sub_model_acc = sum(sub_model_accuracies) / n_est
#
# print('Average accuracy of sub models = %.2f%%' % (ave_sub_model_acc * 100))
#
# voting = VotingClassifier(estimators=sub_estimators, voting='soft')
# voting = voting.fit(W_train, y_train.T.ravel())
# acc = voting.score(W_test, y_test.T.ravel())
# print('Accuracy of voting = %.2f%%' % (acc * 100))


# def bagging(n_estimators, max_samples, verbose=False):
#
#     standard = False
#     # M__pca_ideal = 147
#     # M__lda_ideal = 46
#     #
#     # if verbose:
#     #     print('M__pca_ideal = ', M__pca_ideal)
#     #     print('M__lda_ideal = ', M__lda_ideal)
#
#     M_pca_bag = N - 1
#
#     M_pca = 141  # M__pca_ideal
#     M_lda = 37  # M__lda_ideal
#
#     estimators = [('pca', PCA(n_components=M_pca)), ('lda', LinearDiscriminantAnalysis(n_components=M_lda)),
#                   ('knn', KNeighborsClassifier(n_neighbors=1))]
#
#     base_est = Pipeline(estimators)
#
#     base_est.fit(X_train.T, y_train.T.ravel())
#
#     acc = base_est.score(X_test.T, y_test.T.ravel())
#
#     if verbose:
#         print('Accuracy of base estimator with no pre PCA = %.2f%%' % (acc * 100))
#
#     pca = PCA(n_components=M_pca_bag)
#     W_train = pca.fit_transform(X_train.T)
#     W_test = pca.transform(X_test.T)
#
#     base_est.fit(W_train, y_train.T.ravel())
#     acc = base_est.score(W_test, y_test.T.ravel())
#
#     if verbose:
#         print('Accuracy of base estimator with pre PCA applied = %.2f%%' % (acc * 100))
#
#     estimators = []
#     sub_model_accuracies = []
#
#     W_bag = np.empty((int(max_samples * N), (N - 1)))
#     y_bag = np.empty((1, int(max_samples * N)))
#
#     for i in range(n_estimators):
#
#         for j in range(int(max_samples * N)):
#             mask = np.random.choice(np.arange(N), 1, replace=True)
#             W_bag[j] = W_train[mask, :]
#             y_bag[:, j] = y_train[:, mask]
#
#         estimator = clone(base_est)
#
#         estimator.fit(W_bag, y_bag.T.ravel())
#
#         name = 'est_' + str(i + 1)
#         estimators.append((name, estimator))
#
#         sub_model_acc = estimator.score(W_test, y_test.T.ravel())
#         sub_model_accuracies.append(sub_model_acc)
#         # if verbose:
#         print('Accuracy of sub model ', i + 1, ' = %.2f%%' % (sub_model_acc * 100))
#
#     ave_sub_model_acc = sum(sub_model_accuracies) / n_estimators
#     # if verbose:
#     print('Average accuracy of sub models = %.2f%%' % (ave_sub_model_acc * 100))
#
#     y_hat = []
#
#     for w in W_test:
#         prediction_sum = 0
#         predictions = np.empty(n_estimators, dtype=np.int64)
#         for i, (name, estimator) in enumerate(estimators):
#
#             y = estimator.predict(w.reshape(1, -1))
#             prediction_sum = prediction_sum + float(y[0])
#             predictions[i] = int(y[0])
#         prediction = round(prediction_sum / n_estimators)
#
#         counts = np.bincount(predictions)
#         # y_hat.append(prediction)
#         y_hat.append(np.argmax(counts))
#
#     acc = accuracy_score(y_test.T, y_hat)
#
#     # if verbose:
#     print('Accuracy of ensemble estimator = %.2f%%' % (acc * 100))
#
#     return acc, ave_sub_model_acc, y_hat

#
# n_estimators = 50
# max_samples = 0.8
#
# acc, ave_sub_model_acc = bagging(n_estimators, max_samples)
#
# n_estimators = 30
# max_samples = 0.5
#
# acc_varying_samples = []
# acc_varying_samples_ave = []
# num_samples = []
#
# while max_samples <= 1.0:
#     acc, ave_sub_model_acc = bagging(n_estimators, max_samples)
#     acc_varying_samples.append(acc * 100)
#     acc_varying_samples_ave.append(ave_sub_model_acc * 100)
#     num_samples.append(max_samples)
#     max_samples = max_samples + 0.025
#
# plt.figure(figsize=(8.0, 6.0))
# plt.plot(num_samples, acc_varying_samples, color='mediumseagreen', label='Accuracy of Ensemble Model ')
# plt.plot(num_samples, acc_varying_samples_ave, label='Average Accuracy of Individual Models')
# plt.title('Accuracy versus Ratio of Samples (n_estimators=30)\n')
# plt.xlabel('Ratio of samples')
# plt.ylabel('Recogniton Accuracy [%]')
# plt.legend(loc='best')
# plt.savefig('img/accuracy_versus_ratio_of_samples.png',
#                 dpi=1000, transparent=True)
# plt.show()

# n_estimators = 1
# max_samples = 0.8
#
# acc_varying_num_est_bag = []
# acc_varying_num_est_bag_ave = []
# num_estimators_list = []
# n_est_test_range = 60

# while n_estimators <= n_est_test_range:
#     acc, ave_sub_model_acc = bagging(n_estimators, max_samples)
#     acc_varying_num_est_bag.append(acc * 100)
#     acc_varying_num_est_bag_ave.append(ave_sub_model_acc * 100)
#     num_estimators_list.append(n_estimators)
#     n_estimators = n_estimators + 1
#
#
# plt.figure(figsize=(8.0, 6.0))
# plt.plot(num_estimators_list, acc_varying_num_est_bag, color='mediumseagreen', label='Accuracy of Ensemble Model')
# plt.plot(num_estimators_list, acc_varying_num_est_bag_ave, label='Average Accuracy of Individual Models')
# plt.title('Accuracy versus n_estimators\n')
# plt.xlabel('n_estimators')
# plt.ylabel('Recogniton Accuracy [%]')
# plt.legend(loc='best')
# plt.savefig('img/accuracy_versus_n_estimators.png',
#                 dpi=1000, transparent=True)
# plt.show()


# Finding ideal M_pca and M_lda for bagging
# n_estimators = 30
# max_samples = 1.0
#
# verbose = True
# standard = False
# # M__pca_ideal = 147
# # M__lda_ideal = 46
#
# M_pca_bag = N - 1
#
# M_pca = 1  # M__pca_ideal
# M_lda = 1  # M__lda_ideal
#
# M_pca_ideal_t = None
# M_lda_ideal_t = None
#
# acc_max = 0
#
# while M_pca < 200:
#     M_lda = 1
#     while M_lda < 50:
#
#         estimators = [('pca', PCA(n_components=M_pca)), ('lda', LinearDiscriminantAnalysis(n_components=M_lda)),
#                       ('knn', KNeighborsClassifier(n_neighbors=1))]
#
#         base_est = Pipeline(estimators)
#
#         base_est.fit(X_train.T, y_train.T.ravel())
#         #
#         # acc = base_est.score(X_test.T, y_test.T.ravel())
#         # if verbose:
#         #     print('Accuracy of base estimator with no pre PCA = %.2f%%' % (acc * 100))
#         #
#         pca = PCA(n_components=M_pca_bag)
#         W_train = pca.fit_transform(X_train.T)
#         W_test = pca.transform(X_test.T)
#
#         base_est.fit(W_train, y_train.T.ravel())
#
#         acc1 = base_est.score(W_test, y_test.T.ravel())
#         # if verbose:
#         #     print('Accuracy of base estimator with pre PCA applied = %.2f%%' % (acc * 100))
#
#         estimators = []
#         sub_model_accuracies = []
#
#         W_bag = np.empty((int(max_samples * N), (N - 1)))
#         y_bag = np.empty((1, int(max_samples * N)))
#
#         for i in range(n_estimators):
#
#             for j in range(int(max_samples * N)):
#                 mask = np.random.choice(np.arange(N), 1, replace=True)
#                 W_bag[j] = W_train[mask, :]
#                 y_bag[:, j] = y_train[:, mask]
#
#             estimator = clone(base_est)
#
#             estimator.fit(W_bag, y_bag.T.ravel())
#
#             name = 'est_' + str(i + 1)
#             estimators.append((name, estimator))
#
#             sub_model_acc = estimator.score(W_test, y_test.T.ravel())
#             sub_model_accuracies.append(sub_model_acc)
#             if verbose:
#                 print('Accuracy of sub model ', i + 1, ' = %.2f%%' % (sub_model_acc * 100))
#
#         ave_sub_model_acc = sum(sub_model_accuracies) / n_estimators
#         if verbose:
#             print('Average accuracy of sub models = %.2f%%' % (ave_sub_model_acc * 100))
#
#         y_hat = []
#
#         for w in W_test:
#             prediction_sum = 0
#             predictions = np.empty(n_estimators, dtype=np.int64)
#             for i, (name, estimator) in enumerate(estimators):
#                 y = estimator.predict(w.reshape(1, -1))
#
#                 prediction_sum = prediction_sum + float(y[0])
#                 predictions[i] = int(y[0])
#             prediction = round(prediction_sum / n_estimators)
#
#             counts = np.bincount(predictions)
#             # y_hat.append(prediction)
#             y_hat.append(np.argmax(counts))
#
#         acc = accuracy_score(y_test.T, y_hat)
#         if verbose:
#             print('Accuracy of ensemble estimator = %.2f%%' % (acc * 100))
#
#         if acc > acc_max:
#             acc_max = acc
#             M_pca_ideal_t = M_pca
#             M_lda_ideal_t = M_lda
#
#         M_lda = M_lda + 3
#
#     M_pca = M_pca + 20
#
# print('acc max = ', acc_max, ' for M_pca = ', M_pca_ideal_t, 'and M_lda = ', M_lda_ideal_t)

#acc max =  0.9134615384615384  for M_pca =  141 and M_lda =  37

n_estimators = 30
max_samples = 1.0

verbose = True
standard = False
# M__pca_ideal = 147
# M__lda_ideal = 46

M_pca_bag = N - 1

M_pca = 141  # M__pca_ideal
M_lda = 37  # M__lda_ideal

M_pca_ideal_t = None
M_lda_ideal_t = None

acc_max = 0


estimators = [('pca', PCA(n_components=M_pca)), ('lda', LinearDiscriminantAnalysis(n_components=M_lda)),
                      ('knn', KNeighborsClassifier(n_neighbors=1))]
base_est = Pipeline(estimators)

base_est.fit(X_train.T, y_train.T.ravel())
#
# acc = base_est.score(X_test.T, y_test.T.ravel())
# if verbose:
        #     print('Accuracy of base estimator with no pre PCA = %.2f%%' % (acc * 100))
        #
pca = PCA(n_components=M_pca_bag)
W_train = pca.fit_transform(X_train.T)
W_test = pca.transform(X_test.T)

base_est.fit(W_train, y_train.T.ravel())
acc1 = base_est.score(W_test, y_test.T.ravel())
        # if verbose:
        #     print('Accuracy of base estimator with pre PCA applied = %.2f%%' % (acc * 100))

estimators = []
sub_model_accuracies = []

W_bag = np.empty((int(max_samples * N), (N - 1)))
y_bag = np.empty((1, int(max_samples * N)))

for i in range(n_estimators):

    for j in range(int(max_samples * N)):
        mask = np.random.choice(np.arange(N), 1, replace=True)
        W_bag[j] = W_train[mask, :]
        y_bag[:, j] = y_train[:, mask]

    estimator = clone(base_est)

    estimator.fit(W_bag, y_bag.T.ravel())

    name = 'est_' + str(i + 1)
    estimators.append((name, estimator))

    sub_model_acc = estimator.score(W_test, y_test.T.ravel())
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
        y = estimator.predict(w.reshape(1, -1))

        prediction_sum = prediction_sum + float(y[0])
        predictions[i] = int(y[0])
    prediction = round(prediction_sum / n_estimators)

    counts = np.bincount(predictions)
    # y_hat.append(prediction)
    y_hat.append(np.argmax(counts))

acc = accuracy_score(y_test.T, y_hat)
if verbose:
    print('Accuracy of ensemble estimator = %.2f%%' % (acc * 100))

# if acc > acc_max:
#     acc_max = acc
#     M_pca_ideal_t = M_pca
#     M_lda_ideal_t = M_lda
#
# M_lda = M_lda + 3
# M_pca = M_pca + 20

# print('acc max = ', acc_max, ' for M_pca = ', M_pca_ideal_t, 'and M_lda = ', M_lda_ideal_t)

# classes = set(y_train.ravel())
# acc, ave_sub_model_acc, y_hat = bagging(n_estimators, max_samples)
# cnf_matrix = confusion_matrix(y_test.T, y_hat)
#
# plt.rcParams['figure.figsize'] = [28.0, 21.0]
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True,
#                       title='Nearest Neighbor - Normalized Confusion Matrix',
#                        cmap=plt.cm.Blues)
# plt.show()


