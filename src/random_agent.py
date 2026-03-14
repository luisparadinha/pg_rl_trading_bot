import numpy as np

class random_agent:
    def __init__(self, num_actions):
        self.num_actions = num_actions
        
    def act(self, state):
        return np.random.choice(self.num_actions)