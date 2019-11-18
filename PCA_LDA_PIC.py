# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import seaborn as sns
#sns.set_palette(sns.color_palette("muted"))
#sns.set_style("ticks")
from mpl_toolkits.mplot3d import Axes3D

from PCA import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# KNN Classifer
from sklearn.neighbors import KNeighborsClassifier

# matplotlib backtest for missing $DISPLAY
import matplotlib
# scientific computing library
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
# visualization tools
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8.0, 6.0]
#import seaborn as sns
#sns.set_palette(sns.color_palette("muted"))
#sns.set_style("ticks")
from Visualize import plot_confusion_matrix
# split data preprocessor
from Split import split_data


x = np.linspace(1,51, 51)
y = np.linspace(1, 364, 364)

# X, Y = np.meshgrid(x, y)
# acc_array = np.load('Acc_array.npy')
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.plot_surface(X, Y, acc_array, rstride=1, cstride=1,
#                  cmap='Blues')
# ax.set_title('Accuracy versus $\mathcal{M}$_pca & $\mathcal{M}$_lda\n');
# ax.set_xlabel('$\mathcal{M}$_lda')
# ax.set_ylabel('$\mathcal{M}$_pca')
# ax.set_zlabel('Recognition Accuracy [%]', rotation=180)
# ax.view_init(20, 220)
# plt.savefig('img/accuracy_versus_M_pca&M_lda_new.png', dpi=1000, transparent=True)
#
#
# sns.set()
# f, ax = plt.subplots(figsize=(9, 6))
# sns.heatmap(acc_array.T, fmt="d", cmap='Blues', ax=ax)
#  # plt.setp(label_x, rotation=45, horizontalalignment='right')
# plt.title('Accuracy versus $\mathcal{M}$_pca & $\mathcal{M}$_lda Heatmap\n')
# plt.xticks(rotation=45, horizontalalignment='right')
# plt.yticks(rotation=360, horizontalalignment='right')
# plt.xlabel('$\mathcal{M}$_pca')
# plt.ylabel('$\mathcal{M}$_lda')
# plt.savefig('img/accuracy_versus_M_pca&M_lda_heatmap.png', dpi=1000, transparent=True)
# plt.show()


data = split_data()
X_train, y_train = data['train']
D, N = X_train.shape
X_test, y_test = data['test']
I, K = X_test.shape
_card = 52
D, N = X_train.shape

classes = set(y_train.ravel())

M_pca = 147
M_lda = 46

standard = False

pca = PCA(n_comps=M_pca, standard=standard)
W_train = pca.fit(X_train)
lda = LinearDiscriminantAnalysis(n_components=M_lda)
W_train_2 = lda.fit_transform(W_train.T, y_train.T.ravel())
nn = KNeighborsClassifier(n_neighbors=1)
nn.fit(W_train_2, y_train.T.ravel())
W_test = pca.transform(X_test)
W_test_2 = lda.transform(W_test.T)
acc = nn.score(W_test_2, y_test.T.ravel())
print('M_pca = ', M_pca, ', M_lda = ', M_lda, ' --->  Accuracy = %.2f%%' % (acc * 100))
y_hat = nn.predict(W_test_2)
#cnf_matrix = confusion_matrix(y_test.ravel(), y_hat, labels=list(classes))
#
## plt.rcParams['figure.figsize'] = [28.0, 21.0]
## plt.figure()
## plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True,
##                       title='Nearest Neighbor - Normalized Confusion Matrix',
##                       cmap=plt.cm.Blues)
## plt.savefig('img/pcalda_nn_cnf_matrix.png', dpi=300, transparent=True)
#

SHAPE = (46, 56)

plt.rcParams['figure.figsize'] = [12.0, 9.0]
sns.set_palette(sns.color_palette("muted"))
_palette = sns.color_palette("muted")
sns.set_style("ticks")


done = {'success': False, 'failure': False}
first_failure = True

fig, axes = plt.subplots(nrows=2)

for y, t, w in zip(y_hat, y_test.T.ravel(), W_test.T):
    if y == t and done['success'] is False:
        x_hat = pca.reconstruct(w)
        axes[0].imshow(x_hat.reshape(SHAPE).T,
                       cmap=plt.get_cmap('gray'))
        axes[0].set_title(
            'Successful NN Classification\nPredicted Class: %d, True Class: %d' % (y, t), color=_palette[1])
        done['success'] = True
    elif y != t and done['failure'] is False and first_failure is True:
        first_failure = False
    elif y != t and done['failure'] is False and first_failure is False:
        x_hat = pca.reconstruct(w)
        axes[1].imshow(x_hat.reshape(SHAPE).T,
                       cmap=plt.get_cmap('gray'))
        axes[1].set_title(
            'Failed NN Classification\nPredicted Class: %d, True Class: %d' % (y, t), color=_palette[2])
        done['failure'] = True
    elif done['failure'] is True and done['success'] is True:
        break

fig.tight_layout()
fig.savefig('img/pcalda_nn_class_images.png',
             dpi=1000, transparent=True)
#
#
#
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.decomposition import PCA
#from mpl_toolkits.mplot3d import Axes3D
#
## fig = plt.figure(figsize=plt.figaspect(0.35))
#
## ax = fig.add_subplot(1, 2, 1, projection='3d')
#
#fig = plt.figure(1, figsize=(8, 6))
#ax = Axes3D(fig, elev=-150, azim=110)
#X_reduced = PCA(n_components=3).fit_transform(X_train.T)
#ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y_train.T.ravel(),
#           cmap=plt.cm.Set1, edgecolor='k', s=40)
#ax.set_title("First three PCA dimensions")
#ax.set_xlabel("1st eigenvector")
#ax.w_xaxis.set_ticklabels([])
#ax.set_ylabel("2nd eigenvector")
#ax.w_yaxis.set_ticklabels([])
#ax.set_zlabel("3rd eigenvector")
#ax.w_zaxis.set_ticklabels([])
#plt.savefig('img/first_three_pca.png',
#              dpi=300, transparent=True)
#
#
##ax = fig.add_subplot(1, 2, 2, projection='3d')
#
#fig = plt.figure()
#ax = Axes3D(fig, elev=-150, azim=110)
#X_reduced = LinearDiscriminantAnalysis(n_components=3).fit_transform(X_train.T, y_train.T.ravel())
#ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y_train.T.ravel(),
#           cmap=plt.cm.Set1, edgecolor='k', s=40)
#ax.set_title("First three LDA dimensions")
#ax.set_xlabel("1st eigenvector")
#ax.w_xaxis.set_ticklabels([])
#ax.set_ylabel("2nd eigenvector")
#ax.w_yaxis.set_ticklabels([])
#ax.set_zlabel("3rd eigenvector")
#ax.w_zaxis.set_ticklabels([])
#plt.savefig('img/first_three_lda.png',
#              dpi=300, transparent=True)
#
#
#plt.show()