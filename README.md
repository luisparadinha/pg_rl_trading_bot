# RL Trading Bot

## Overview
Brief description of the project — what it does, what problem it solves.

## Agents
- **Random** — baseline, uniform random actions
- **Momentum** — buys/sells based on price trend over a window
- **Buy & Hold** — buys on day 1 and holds until the end
- **Q-Learning** — tabular Q-learning with several state representations

## State Representations
- Raw state (prices, holdings, cash)
- Discretized volatility + holdings
- Discretized volatility + momentum + holdings

## Results
Brief summary of findings — e.g. Buy & Hold dominates on a bull market, Q-Learning with discretized state shows learning but doesn't beat passive strategies.

## Notebook
[View notebook](https://github.com/luisparadinha/pg_rl_trading_bot/blob/master/Market_e_Projecto.ipynb)

## Requirements
```bash
pip install -r requirements.txt