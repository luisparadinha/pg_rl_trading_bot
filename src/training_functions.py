def play_episode(agent, market):
    state = market.start()
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done = market.new_day(action)
        state = next_state

    return market.get_episode_value()


def play_episode_sequence(agent, market, n_episodes):
    """
    Runs n_episodes episodes and returns a list of final portfolio values.
    """
    values_list = []
    for episode in range(n_episodes):
        value = play_episode(agent, market)
        values_list.append(value)
    return values_list