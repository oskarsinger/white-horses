import numpy as np

class BatchGramServer:

    def __init__(self, reg=10**(-5)):
        
        self.reg = reg

    def get_gram(self, batch):

        gram = np.dot(batch.T, batch)

        gram += np.eye(batch.shape[1]) * self.reg

        return gram

