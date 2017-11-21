import numpy as np

def get_random_k_folds(N, k):

    indexes = np.arange(N)
    
    np.random.shuffle(indexes)

    size = int(N / k)
    holdouts = [indexes[size*i:size*(i+1)]
                for i in range(k)]
    folds = [np.hstack(holdouts[:i] + holdouts[i+1:])
             for i in range(k)]

    return list(zip(folds, holdouts))
    
def get_one_hots(A):

    N = A.shape[0]
    unique = np.unique(A)
    num_unique = unique.shape[0]
    one_hots = np.zeros((N, num_unique))
    
    for (i, u) in enumerate(unique):
        one_hots[A == u, i] = 1

    return one_hots

def get_minibatch(A, batch_size):

    indexes = np.random.choice(
        A.shape[0],
        replace=False, 
        size=batch_size)

    return A[indexes,:]

