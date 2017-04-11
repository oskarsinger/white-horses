import numpy as np

#TODO: cite Tim's code/paper here
class SyntheticHeartStepsLoader:

    def __init__(self, 
        age=None, 
        gender=None,
        activity=None):

        if age is None:
            age = np.random.poisson(lam=1)

        self.age = age

        if gender is None:
            gender = np.random.binomial(1, 0.351)

        self.gender = gender

        if activity is None:
            activity = np.random.normal(scale=2.3)

        self.activity = activity
        self.baseline = np.array([
            self.age,
            self.gender,
            self.activity])[:,np.newaxis]

        mean_w = np.array([
            1.462, 
            0.362, 
            0.3, 
            0.003, 
            0.231, 
            -0.038, 
            0.206, 
            0.197, 
            -1.126, 
            0.007, 
            0.368, 
            0.079, 
            0.002, 
            -0.033, 
            -0.02, 
            -0.004, 
            -0.354, 
            0.273, 
            -0.006, 
            -0.034, 
            0.001, 
            0.011, 
            -0.091, 
            -0.205])[:,np.newaxis]

        self.w = np.random.randn(mean_w.shape[0], 1) + mean_w
        self.actions = []
        self.rewards = []
        self.num_rounds = 0

    def get_data(self):

        self.num_rounds += 1

        pre_steps = np.random.normal(loc=2.7, scale=3.03)
        engaged_level = np.random.uniform()

        self.current_state = np.vstack([
            pre_steps, 
            engaged_level, 
            self.baseline])

        return self.current_state

    def set_action(self, action):

        self.actions.append(action) 

        self.current_covariates = np.vstack([
            1,
            self.current_state,
            action * self.current_state,
            self.current_state[0] * self.baseline,
            self.current_state[1] * self.baseline,
            self.current_state[0] * self.baseline * action,
            self.current_state[1] * self.baseline * action])

    def get_reward(self):

        mean = np.dot(self.w.T, self.current_covariates)
        reward = np.random.normal(
            loc=mean,
            scale=np.sqrt(7.47))

        self.rewards.append(reward)

        return reward

    def name(self):

        return 'SyntheticHeartStepsLoader'

    def cols(self):

        return self.w.shape[0]

    def rows(self):

        return self.num_rounds
