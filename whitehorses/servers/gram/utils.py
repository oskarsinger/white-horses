import numpy as np

def get_gram(A, reg=None):

    gram = np.dot(A.T, A)

    reg_matrix = None

    if reg is not None:
        reg_matrix = reg \
            if np.isscalar(gram) else \
            reg * np.identity(gram.shape[0])
        gram = gram + reg_matrix

    return gram

