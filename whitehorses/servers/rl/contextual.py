import numpy as np

class ContextualBanditServer:

    def __init__(self, loader):

        self.loader = loader

        self.num_rounds = 0

    def get_data(self):

        self.num_rounds += 1

        return self.loader.get_data()

    def set_action(self, action):

        self.loader.set_action(action)

    def get_reward(self):

        return self.loader.get_reward()

    def cols(self):

        return self.loader.cols()

    def rows(self):

        return self.loader.rows()

    def name(self):

        return 'BanditServer'
