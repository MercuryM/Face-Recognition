from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import itertools
from Split import split_data
from sklearn.metrics import confusion_matrix
from Visualize import plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.base import clone
from sklearn.metrics import accuracy_score
import seaborn as sns
sns.set_palette(sns.color_palette("muted"))
sns.set_style("ticks")

seed = 7
data = split_data()
X_train, y_train = data['train']
D, N = X_train.shape
X_test, y_test = data['test']
I, K = X_test.shape

classes = set(y_train.ravel())

M0_ideal = 150
M1_ideal = 80

n_estimators = 30
M0 = M0_ideal
M1 = M1_ideal

verbose = True

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
assert (M0 + M1 > M_pca)
assert (M_pca > M_lda)

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
    prediction = round(prediction_sum / n_estimators)

    counts = np.bincount(predictions)
    # y_hat.append(prediction)
    y_hat.append(np.argmax(counts))

acc = accuracy_score(y_test.T, y_hat)
if verbose:
    print('Accuracy of ensemble estimator = %.2f%%' % (acc * 100))

cnf_matrix = confusion_matrix(y_test.T, y_hat)

class_names = np.arange(1, 53)

plt.figure()

plt.rcParams['figure.figsize'] = [28.0, 21.0]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True,
                       title='Majority voting - Normalized Confusion Matrix',
                       cmap=plt.cm.Blues)

plt.show()
plt.savefig('img/ensemble_nn_cnf_matrix_final_voting.png', dpi=300, transparent=True)

# pca1 = PCA(n_components=147)
# lda = LinearDiscriminantAnalysis(n_components=46)
#
# W_train = pca1.fit(X_train)
# W_train_2 = lda.fit_transform(W_train.T, y_train.T.ravel())
# W_test = pca1.transform(X_test.T)
# W_test_2 = lda.transform(W_test.T)
# X_hat = lda.inverse_transform(W_test_2)
#
# col_index = 0
#
# # prettify plots
# plt.rcParams['figure.figsize'] = [12.0, 9.0]
# sns.set_palette(sns.color_palette("muted"))
# _palette = sns.color_palette("muted")
# sns.set_style("ticks")
#
# fig, axes = plt.subplots(ncols=3, nrows=2)
#
# for y, y_, x_, x in zip(y_hat, y_test.T.ravel(), X_hat, X_test.T):
#     if (y != y_):
#         axes[0, col_index].imshow(x.reshape(46, 56).T,
#                                   cmap=plt.get_cmap('gray'))
#         axes[0, col_index].set_title(
#             'Predicted Class: %d, True Class: %d' % (y, y_) + '\nOriginal Image', color=_palette[2])
#         axes[1, col_index].imshow(x_.reshape(46, 56).T,
#                                   cmap=plt.get_cmap('gray'))
#         axes[1, col_index].set_title(
#             'Reconstructed Image', color=_palette[2])
#         col_index = col_index + 1
#
# fig.tight_layout()
# fig.savefig('img/ensemble_nn_class_images.png',
#               dpi=300, transparent=True)