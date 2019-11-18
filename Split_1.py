#import numpy as np
#import scipy.io
from Split_2 import split_subset

def split_4_subsets(training_X, training_l):
    
    data = split_subset(training_X,training_l)
    first_split1_X, first_split1_l = data['train']
    #D, training_num = first_split_X.shape
    first_split2_X, first_split2_l = data['test']
    #I, K = testing_X.shape
    
    data_1 = split_subset(first_split1_X,first_split1_l)
    second_split1_X, second_split1_l = data_1['train']
    #D, training_num = first_split_X.shape
    second_split2_X, second_split2_l = data_1['test']
    
    data_2 = split_subset(first_split2_X,first_split2_l)
    second_split3_X, second_split3_l = data_2['train']
    #D, training_num = first_split_X.shape
    second_split4_X, second_split4_l = data_2['test']
    
    
    
    return {'subset1': (second_split1_X, second_split1_l), 'subset2': (second_split2_X, second_split2_l),'subset3': (second_split3_X, second_split3_l), 'subset4': (second_split4_X, second_split4_l)}
    

#    """
#    Function for splitting data.
#    Parameters
#    ----------
#    fname: str
#        Name of the '.mat' input file
#    ratio: float
#        Split ratio of dataset
#    seed: int
#        Random seed initial state
#    Returns
#    -------
#    data: dict
#        * train: tuple
#            - X: features
#            - y: labels
#        * test: tuple
#            - X: features
#            - y: labels
#    """
#
#    # load '.mat' file
#    #data = scipy.io.loadmat(fname)
#
#    # Images
#    # N: number of images
#    # D: number of pixels
#    X = training_X  # shape: [D x N]
#    y = training_l  # shape: [1 x N]
#    assert(X.shape[1] == y.shape[1])
#
#    # Number of images
#    D, N = X.shape
#    # Fix the random seed
#    np.random.seed(seed)
#    # Cardinality of labels
#    _card = len(set(y.ravel()))
#    # Step splitting of dataset
#    _step = int(N / _card)
#    # Shape boundaries
#    _bounds = np.arange(0, N+1, _step)
#    # Shapes
#    shapes = list(zip(_bounds[:-1], _bounds[1:]))
#
#   # Train set
#    _mask = []
#    for _shape in shapes:
#        _idx = np.random.choice(np.arange(*_shape), int(ratio * _step), replace=False)
#        _mask.append(_idx)
#    mask_train = np.array(_mask).ravel()
#    mask_test = np.array(list(set(np.arange(0, N)).difference(set(mask_train))))
#
#    # Partition dataset to train and test sets
#    X_train, X_test = X[:, mask_train], X[:, mask_test]
#    y_train, y_test = y[:, mask_train], y[:, mask_test]
#
#    return {'train': (X_train, y_train), 'test': (X_test, y_test)}