class buy_hold_agent:
    def __init__(self, num_companies):
        # store how many companies exist in the market
        self.num_companies = num_companies
        # flag to track if this is the first step of the episode
        self.first_step = True

    def act(self, state):
        # on the first step, buy everything
        if self.first_step:
            # mark that we have already taken the first step
            self.first_step = False
            # build the "buy all" action index
            # actions are encoded in base 3: 0=sell, 1=hold, 2=buy
            # for each company i, add 2 * 3^i to get the index for "buy all"
            action = 0
            for i in range(self.num_companies):
                action += 2 * (3 ** i)
            return action

        # on all subsequent steps, hold everything
        # for each company i, add 1 * 3^i to get the index for "hold all"
        action = 0
        for i in range(self.num_companies):
            action += 1 * (3 ** i)
        return action

    def reset(self):
        # reset the flag so the agent buys again at the start of a new episode
        self.first_step = True