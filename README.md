# RL Trading Bot

A small reinforcement learning project that explores trading strategies on historical stock data. The repository contains experiments with baseline rules-based agents and early-stage learning-based agents, plus a Jupyter notebook for analysis and results.

## Project Overview

This project uses historical price data for selected stocks to simulate trading performance and compare different agent strategies.

Key components:
- rule-based baselines: Random, Momentum, Buy & Hold
- exploratory reinforcement learning: tabular Q-Learning with discrete state representations
- analysis notebook: `Market_e_Projecto.ipynb`
- sample data files: `data/AAPL.csv`, `data/AMZN.csv`, `data/GOOGL.csv`, `data/MSFT.csv`, `data/NOK.csv`

## Repository Structure

- `Market_e_Projecto.ipynb` — exploratory notebook with strategy implementation, simulation, and results
- `requirements.txt` — project dependencies
- `data/` — historical market data CSV files used for experiments
- `src/` — Python source code and supporting modules

## Getting Started

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Open the notebook:

```bash
jupyter notebook Market_e_Projecto.ipynb
```

## Usage

- Run the notebook to explore the trading agents and review simulation results.
- Use the data files in `data/` for backtesting and analysis.
- Extend the project by adding new RL agents, richer state features, or a stronger market simulator.

## Notes

- The current implementation is experimental and intended for academic exploration rather than production trading.
- Results depend on the data in `data/` and the choice of strategy hyperparameters.

## License

This repository does not include a license file. Add one if you want to specify reuse terms.

