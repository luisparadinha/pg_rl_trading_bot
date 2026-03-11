import numpy as np

class MomentumAgent:
    def __init__(self, num_actions, num_companies, window=3):
        self.num_actions = num_actions
        self.num_companies = num_companies
        self.window = window
        self.history = []

    def act(self, state):
        prices = state[self.num_companies:2*self.num_companies] # slicer para obter apenas os preços. Vai desde o numero de empresas até o dobro do número de empresas, assumindo que o estado é estruturado como [holdings, prices]
        self.history.append(prices.copy()) # adiciona os preços atuais ao histórico. Usa copy para evitar que futuras modificações em prices afetem o histórico

        if len(self.history) < self.window + 1: # se ainda não houver histórico suficiente para calcular o momentum, escolhe uma ação aleatória
            return np.random.choice(self.num_actions)

        # Calculate momentum
        momentum = self.history[-1] - self.history[-self.window-1] # calcula o momentum como a diferença entre o preço atual e o preço de n passos atrás (definido pela janela)
        actions = []

        for m in momentum:
            if m > 0:
                actions.append(2)  # Buy
            elif m < 0:
                actions.append(0)  # Sell
            else:
                actions.append(1)  # Hold

        # Convert to index
        action_index = 0
        for i, a in enumerate(actions):
            action_index += a * (3 ** (self.num_companies - i - 1))

        return action_index
