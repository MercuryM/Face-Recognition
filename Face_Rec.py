# matplotlib backtest for missing $DISPLAY
import matplotlib
matplotlib.use('Agg')
# scientific computing library
import numpy as np
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
# utility functions
from Utils import progress

# built-in tools
import time
import os
import psutil

if __name__ == '__main__':

    n_neighbors =  1
    data = split_data()
    X_train, y_train = data['train']
    D, N = X_train.shape
    X_test, y_test = data['test']
    I, K = X_test.shape

    # mean face
    mean_face = X_train.mean(axis=1).reshape(-1, 1)
    A = X_train - mean_face
    S = (1 / N) * np.dot(A.T, A)

    # Calculate eigenvalues 'w' and eigenvectors 'v'
    _l, _v = np.linalg.eig(S)
    # Sorted eigenvalues and eigenvectors
    _indexes = np.argsort(_l)[::-1]
    l = _l[_indexes]
    v = _v[:, _indexes]

    classes = set(y_train.ravel())
    Ms = np.arange(1, N + 1)

    # value of M for confusion matrix
    M_star = Ms[-1]
    acc = []
    train_dur = []
    test_dur = []
    memory = []

    for j, M in enumerate(Ms):

        progress(j + 1, len(Ms), status='Model for M=%d' % M)

        # start timer
        _start = time.time()
        V = v[:, :M]
        _U = np.dot(A, V)
        U = _U / np.apply_along_axis(np.linalg.norm, 0, _U)
        W = np.dot(U.T, A)
        classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
        classifier.fit(W.T, y_train.T.ravel())
        # stop timer
        _stop = time.time()
        # train time
        train_dur.append(_stop - _start)

        assert I == D, print(
            'Number of features of test and train data do not match!')
        accuracy = 0
        # start timer
        _start = time.time()
        Phi = X_test - mean_face
        W_test = np.dot(U.T, Phi)
        W_test = np.dot(Phi.T, U)
        y_hat = classifier.predict(W_test)
        accuracy = 100 * np.sum(y_test.ravel() == y_hat) / K
        # stop timer
        _stop = time.time()
        # test time
        test_dur.append(_stop - _start)

        # store values for confusion matrix
        if M == M_star:
            cnf_matrix = confusion_matrix(
                y_test.ravel(), y_hat, labels=list(classes))

        # pct memory usage
        memory.append(psutil.Process(os.getpid()).memory_percent())

        acc.append(accuracy)

    print('\033[1;32m')
    print('Best Accuracy = %.2f%%' % (np.max(acc)))

    plt.plot(Ms, acc)
    plt.title(
        'Recognition Accuracy versus $\mathcal{M}$\n')
    plt.xlabel('$\mathcal{M}$: number of principle components')
    plt.ylabel('Recognition Accuracy [%]')
    plt.savefig('img/accuracy_versus_M.png',
                dpi=1000, transparent=True)

    plt.figure()
    plt.plot(Ms, train_dur)
    #sns.regplot(x=Ms.reshape(-1, 1), y=np.array(train_dur))
    plt.plot(Ms, test_dur)
    #sns.regplot(x=Ms.reshape(-1, 1), y=np.array(test_dur))
    plt.title('Execution Time versus $\mathcal{M}$\n')
    plt.xlabel('$\mathcal{M}$: number of principle components')
    plt.ylabel('Execution Time [sec]')
    plt.legend(['Train', 'Test'])
    plt.savefig('img/time_versus_M.png',
                 dpi=1000, transparent=True)

    plt.figure()
    plt.plot(Ms, memory)
    # sns.regplot(x=Ms.reshape(-1, 1), y=np.array(memory))
    plt.title('Memory Percentage Usage versus $\mathcal{M}$\n')
    plt.xlabel('$\mathcal{M}$: number of principle components')
    plt.ylabel('Memory Usage [%]')
    plt.savefig('img/memory_versus_M.png',
                dpi=1000, transparent=True)


    plt.rcParams['figure.figsize'] = [28.0, 21.0]

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=classes,
                          title='Nearest Neighbor - Confusion Matrix',
                          cmap=plt.cm.Blues)
    plt.savefig('img/nn_cnf_matrix.png', dpi=300, transparent=True)
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True,
                          title='Nearest Neighbor - Normalized Confusion Matrix',
                          cmap=plt.cm.Blues)
    plt.savefig('img/nn_cnf_matrix_norm.png', dpi=300, transparent=True)

