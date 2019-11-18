import numpy as np
import matplotlib.pyplot as plt
from Split import split_data
from PCA import PCA
from Utils import progress
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# KNN Classifer
from sklearn.neighbors import KNeighborsClassifier
# ignore warnings
import warnings
#import seaborn as sns
#sns.set_palette(sns.color_palette("muted"))
#sns.set_style("ticks")
warnings.filterwarnings("ignore")

from mpl_toolkits import mplot3d

data = split_data()
X_train, y_train = data['train']
D, N = X_train.shape
X_test, y_test = data['test']
I, K = X_test.shape

_card = 52

M_pca = 1
M_lda = 1

M_pca_range = N - _card
M_lda_range = _card - 1

acc_array = np.empty((M_pca_range , M_lda_range))
M_pca_array = np.arange(1, M_pca_range + 1)
M_lda_array = np.arange(1, M_lda_range + 1)

standard = False

M__pca_ideal = None
M__lda_ideal = None
acc_max = 0

Ms = np.arange(1, (M_pca_range) * (M_lda_range) + 1)

while M_lda <= M_lda_range:

    M_pca = M_lda

    while  M_pca <= M_pca_range:

#         for j in range((M_pca - 1) * 51, M_pca * 51):
#             progress(j + 1, len(Ms), status = 'Model for M_pca=%d, M_lda=%d' %(M_pca, M_lda))
             pca = PCA(n_comps = M_pca, standard = standard)
             W_train = pca.fit(X_train)
             lda = LinearDiscriminantAnalysis(n_components = M_lda)
             W_train_2 = lda.fit_transform(W_train.T, y_train.T.ravel())
             nn = KNeighborsClassifier(n_neighbors = 1)
             nn.fit(W_train_2, y_train.T.ravel())
             W_test = pca.transform(X_test)
             W_test_2 = lda.transform(W_test.T)
             acc = nn.score(W_test_2, y_test.T.ravel())
             acc_array[M_pca - 1, M_lda - 1] = acc
             print('M_pca = ', M_pca, ', M_lda = ', M_lda, ' --->  Accuracy = %.2f%%' % (acc * 100))
             if (acc > acc_max):
                M__pca_ideal = M_pca
                M__lda_ideal = M_lda
                acc_max = acc
             M_pca = M_pca + 1
    M_lda = M_lda + 1
print("Accuracy is maximum for M__pca = ", M__pca_ideal, ", M_lda = ", M__lda_ideal,
      " with accuracy of %.2f%%" % (acc_max * 100), ".")

# Ideal: M_pca =  147 , M_lda =  46  --->  Accuracy = 94.23%

x = np.linspace(1, M_lda_range, M_lda_range)
y = np.linspace(1, M_pca_range, M_pca_range)

X, Y = np.meshgrid(x, y)

print(acc_array.shape)
print(X.shape)
print(Y.shape)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, acc_array, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('Accuracy varying M_pca & M_lda');
ax.set_xlabel('M_lda')
ax.set_ylabel('M_pca')
ax.set_zlabel('Accuracy');

ax.view_init(30, 220)
plt.savefig('img/accuracy_versus_M_pca&M_lda.png', dpi=1000, transparent=True)