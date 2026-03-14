import numpy as np

class qlearning_agent:
    def __init__(self, 
                num_actions, 
                num_companies, 
                alpha=0.1, 
                gamma=0.95, 
                epsilon=1.0, 
                epsilon_decay=0.995, 
                epsilon_min=0.01
                ):
    
        # ===== General RL agent attributes =====

        # save number of actions for later use
        self.num_actions = num_actions

        # save number of companies for later use
        self.num_companies = num_companies

        # learning rate: how much we update Q values on each step
        self.alpha = alpha

        # discount factor: how much we value future rewards vs immediate rewards
        self.gamma = gamma

        # exploration rate: probability of choosing a random action
        self.epsilon = epsilon

        # decay rate for epsilon: how quickly epsilon decreases towards epsilon_min
        self.epsilon_decay = epsilon_decay

        # minimum exploration rate: lowest value epsilon can decay to
        self.epsilon_min = epsilon_min

        # ===== Q-learning specific attributes =====

        # Q-table: dictionary mapping state -> array of Q values, one per action
        # equivalent to Q[s][a] in the teacher's code
        # we use a dict because the state space is too large to pre-build (unlike grid world)
        self.Q = {}

    # ===== Internal helper methods =====

    def _state_key(self, state): # _ before method name indicates it's intended to be internal/private
        # states are numpy arrays, which can't be dict keys, so we convert them to tuples
        return tuple(state)

    def _get_q_values(self, state_key): # _ before method name indicates it's intended to be internal/private
        # if we've never seen this state before, initialize Q values to zero for all actions
        # equivalent to Q[s] = np.zeros(num_actions) for Grid World problem
        if state_key not in self.Q:
            self.Q[state_key] = np.zeros(self.num_actions)
        return self.Q[state_key]
    
    # ===== Main agent methods =====
    
    def act(self, state):
        # epsilon-greedy: explore with probability epsilon, otherwise exploit (choose action with highest Q value)
        # equivalent to epsilon_greedy(s, eps) in Grid World problem
        state_key = self._state_key(state)
        self._get_q_values(state_key) # initialize Q values if state is new

        if np.random.random() < self.epsilon:
            return np.random.choice(self.num_actions) # explore: choose a random action
        else:
            return np.argmax(self.Q[state_key]) # exploit: choose action with highest Q value
        
    def learn(self, state, action, reward, next_state, done):
        # q-learning update rule:
        state_key = self._state_key(state)
        next_state_key = self._state_key(next_state)

        current_q = self._get_q_values(state_key)[action]
        next_q_values = self._get_q_values(next_state_key)

        if done:
            # if episode is done, there's no future reward to consider
            target = reward
        else:
            # bootstrapping: estimate future value using next best action
            target = reward + self.gamma * np.max(next_q_values)

        # update q value for this (state, action) pair
        self.Q[state_key][action] += self.alpha * (target - current_q)

        # decay epsilon after each learning step
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
