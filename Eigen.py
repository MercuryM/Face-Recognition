import numpy as np
import matplotlib.pyplot as plt
import time

SHAPE = (46, 56)
# split data preprocessor
from Split import split_data

data=split_data( 'face' , 0.8, 13);
X_train, Y_train=data['train']
X_test, Y_test=data['test']

print(data)

# mean face
mean_face = X_train.mean(axis=1).reshape(-1, 1)
# plt.imshow(mean_face.reshape(SHAPE).T,
#            cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
# plt.title('Mean Face\n')
# plt.savefig('img/mean_face.png', dpi=1000, transparent=True)

D, N = X_train.shape
A = X_train - mean_face

t = time.time()
# high dimension
# S = (1 / N) * np.dot(A, A.T)
# low dimension
S = (1 / N) * np.dot(A.T, A)

# Calculate eigenvalues 'w' and eigenvectors 'v'
_w, _u = np.linalg.eig(S)
# Calculating time
print('Duration %.2fs' % (time.time() - t))
# Sorted eigenvalues and eigenvectors
_indexes = np.argsort(np.abs(_w))[::-1]
w = _w[_indexes]
u = np.real(_u[:, _indexes])
#
plt.figure(figsize=(8.0, 6.0))
plt.bar(range(len(w)), np.abs(w))
plt.title('Sorted Eigenvalues')
plt.xlabel('$w_{m}$: $m^{th}$ eigenvalue')
plt.ylabel('Real Value')
# plt.savefig('img/eigenvalues_h.png', dpi=1000, transparent=True)
plt.savefig('img/eigenvalues_l.png', dpi=1000, transparent=True)
print()

# n_images = 3
# fig, axes = plt.subplots(nrows=1, ncols=n_images)
# for ax, img, i in zip(axes.flatten(), u[:, :n_images].T, range(1, n_images + 1)):
#         ax.imshow(img.reshape(46,56).T,
#                   cmap=plt.cm.Greys)
#         ax.set_title('Eigenface %d' % i)
# fig.tight_layout()
# plt.savefig('img/Eigenfaces.png', dpi=1000, transparent=True)