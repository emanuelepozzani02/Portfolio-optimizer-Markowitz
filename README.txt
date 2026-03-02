# 🇮🇹 Italian Elite Portfolio Optimizer
**Quantitative Finance project focused on the Italian "National Champions" (LDO.MI, ENI.MI, RACE.MI, G.MI, A2A.MI).**

## 🚀 Project Overview
This tool implements a **Mean-Variance Optimization (Markowitz)** framework to build an optimal equity portfolio using historical data from the FTSE MIB.

### Key Features:
- **Efficient Frontier**: 20,000 Monte Carlo simulations to find the Max Sharpe Ratio.
- **Risk Metrics**: Implementation of **Sortino Ratio**, **Calmar Ratio**, and **Ulcer Index**.
- **Price Projection**: 1-year forward-looking simulation using **Geometric Brownian Motion (GBM)** with Itô's Correction.
- **Backtesting**: Comparative analysis against the **FTSE MIB** benchmark.

## 🛠️ How to run
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the script: `python your_script_name.py`