# scientific computing library
import numpy as np
# matplotlib backtest for missing $DISPLAY
import matplotlib
matplotlib.use('Agg')
# visualization tools
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = [8.0, 6.0]
# split data preprocessor
from Split import split_data
# utility functions
from Utils import progress
from PCA import PCA

SHAPE = (46, 56)

if __name__ == '__main__':

    data = split_data()
    X_train, y_train = data['train']
    D, N = X_train.shape
    X_test, y_test = data['test']
    I, K = X_test.shape

    # # mean face
    mean_face = X_train.mean(axis=1).reshape(-1, 1)
    A = X_train - mean_face
    S = (1 / N) * np.dot(A.T, A)

    # Calculate eigenvalues 'w' and eigenvectors 'v'
    _l, _v = np.linalg.eig(S)
    # Sorted eigenvalues and eigenvectors
    _indexes = np.argsort(_l)[::-1]
    l = _l[_indexes]
    v = _v[:, _indexes]

    M = np.arange(1, N)
    error = []
    for j, m in enumerate(M):

        progress(j + 1, len(M), status='Reconstruction for M=%d' % m)
        V = v[:, :m]
        _U = np.dot(A, V)
        U = _U / np.apply_along_axis(np.linalg.norm, 0, _U)
        W = np.dot(U.T, A)
        A_hat = np.dot(U, W)
        error.append(np.mean(np.linalg.norm(A - A_hat, axis = 1)))
    print('')

    plt.figure(figsize=(8.0, 6.0))
    plt.bar(M, np.array(error))
    plt.title('Reconstruction Error versus Number of Principle Components $\mathcal{M}$\n')
    plt.xlabel('$\mathcal{M}$: Number of Principle Components')
    plt.ylabel('$\mathcal{J}$: Reconstruction Error')
    plt.savefig('img/error_versus_M.png',
                dpi=1000, transparent=True)
    #
    # # examples
    # # set M
    # m = 100
    # V = v[:, :m]
    # _U = np.dot(A, V)
    # U = _U / np.apply_along_axis(np.linalg.norm, 0, _U)
    #
    # # train data
    # W_train = np.dot(U.T, A)
    # print (W_train)
    # rad_train = np.random.randint(0, N, 3)
    # R_train = W_train[:, rad_train]
    # B_train = np.dot(U, R_train)
    #
    # # test data
    # assert I == D, print(
    #     'Number of features of test and train data do not match, %d != %d' % (D, I))
    # Phi = X_test - mean_face
    # W_test = np.dot(U.T, Phi)
    # rad_test = np.random.randint(0, K, 3)
    # R_test = W_test[:, rad_test]
    # B_test = np.dot(U, R_test)

    # plt.rcParams['figure.figsize'] = [16.0, 12.0]
    # fig, axes = plt.subplots(nrows=2, ncols=3)
    # titles_train = ['Original Train', 'Original Train', 'Original Train',
    #                 'Reconstructed Train', 'Reconstructed Train', 'Reconstructed Train']
    # for ax, img, title in zip(axes.flatten(), np.concatenate((A[:, rad_train], B_train), axis=1).T, titles_train):
    #     _img = img + mean_face.ravel()
    #     ax.imshow(_img.reshape(SHAPE).T,
    #               cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
    #     ax.set_title(title)
    # fig.savefig('img/reconstructed_train_images.png',
    #             dpi=1000, transparent=True)
    # print((A[:, rad_train], B_train))
    #
    # fig, axes = plt.subplots(nrows=2, ncols=3)
    # titles_test = ['Original Test', 'Original Test', 'Original Test',
    #                'Reconstructed Test', 'Reconstructed Test', 'Reconstructed Test']
    # for ax, img, title in zip(axes.flatten(), np.concatenate((Phi[:, rad_test], B_test), axis=1).T, titles_test):
    #     _img = img + mean_face.ravel()
    #     ax.imshow(_img.reshape(SHAPE).T,
    #               cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
    #     ax.set_title(title)
    # fig.savefig('img/reconstructed_test_images.png',
    #              dpi=1000, transparent=True)

    # plt.rcParams['figure.figsize'] = [12.0, 9.0]
    # sns.set_palette(sns.color_palette("muted"))
    # _palette = sns.color_palette("muted")
    # sns.set_style("ticks")
    #
    # fig, axes = plt.subplots(ncols=7)
    #
    # axes[0].imshow(X_test.T[7].reshape(46, 56).T, cmap=plt.get_cmap('gray'))
    # axes[0].set_title('Original Face', color=_palette[0])
    #
    # M = 10
    # i = 1
    # j = 1
    #
    # while M < 416:
    #     standard = False
    #
    #     pca = PCA(n_comps=M, standard=standard)
    #
    #     W_train = pca.fit(X_train)
    #
    #     I, K = X_test.shape
    #
    #     W_test = pca.transform(X_test)
    #
    #     x_hat = pca.reconstruct(W_test.T[7])
    #     axes[i].imshow(x_hat.reshape(46, 56).T, cmap=plt.get_cmap('gray'))
    #     axes[i].set_title('M =' + str(M), color=_palette[j])
    #
    #     M = M * 2
    #     i = i + 1
    #     j = j + 1
    # plt.savefig('img/reconstructed_faces_versus_M.png', dpi=1000, transparent=True)