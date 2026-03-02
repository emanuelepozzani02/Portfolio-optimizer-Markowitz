import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# INITIAL SETTINGS
# ============================================================
np.random.seed(42)
tickers = ["LDO.MI", "A2A.MI", "ENI.MI", "G.MI", "RACE.MI"]
benchmark_ticker = "FTSEMIB.MI"
rf = 0.02  # Risk-Free Rate (e.g., 2.0%)

print("Downloading historical data...")
data = yf.download(tickers, period="3y")["Close"]
bench_data = yf.download(benchmark_ticker, period="3y")["Close"]

data = data.dropna()
bench_data = bench_data.dropna()

# ============================================================
# 1. RETURNS CALCULATION
# ============================================================
# SIMPLE Returns: Used for weights, expected returns, and backtesting
simple_returns = data.pct_change().dropna()
# LOG Returns: Used for correlations, volatility, and GBM simulations
log_returns = np.log(data / data.shift(1)).dropna()

# Optimization Parameters (using SIMPLE returns for Markowitz)
avg_returns = simple_returns.mean() * 252
cov_matrix = simple_returns.cov() * 252

# ============================================================
# 2. MONTE CARLO SIMULATION FOR OPTIMAL WEIGHTS
# ============================================================
num_portfolios = 20000
results = np.zeros((3, num_portfolios))
weights_record = []

for i in range(num_portfolios):
    weights = np.random.random(len(tickers))
    weights /= np.sum(weights)
    weights_record.append(weights)

    portfolio_return = np.sum(avg_returns * weights)
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    results[0, i] = portfolio_return
    results[1, i] = portfolio_std_dev
    results[2, i] = (results[0, i] - rf) / results[1, i]  # Sharpe Ratio

# ============================================================
# 3. IDENTIFYING THE OPTIMAL PORTFOLIO
# ============================================================
max_sharpe_idx = np.argmax(results[2])
best_w = weights_record[max_sharpe_idx]

# ============================================================
# CHART: ASSET ALLOCATION (DONUT CHART)
# ============================================================
plt.figure(figsize=(8, 8))
num_tickers = len(tickers)
cmap = plt.colormaps['tab10']
colors = [cmap(i) for i in np.linspace(0, 1, num_tickers)]

plt.pie(best_w, labels=tickers, autopct='%1.1f%%', startangle=90,
        colors=colors, pctdistance=0.85)
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.title("Optimal Asset Allocation (Max Sharpe Ratio)")
plt.tight_layout()
plt.show()

# ============================================================
# 4. EFFICIENT FRONTIER + BENCHMARK
# ============================================================
plt.figure(figsize=(12, 7))
plt.scatter(results[1, :], results[0, :], c=results[2, :],
            cmap='RdYlGn', marker='o', s=10, alpha=0.3)
plt.colorbar(label='Sharpe Ratio')
plt.scatter(results[1, max_sharpe_idx], results[0, max_sharpe_idx],
            color='red', marker='*', s=300, label='Optimal Portfolio', zorder=5)

bench_ret_simple = bench_data.pct_change().mean() * 252
bench_vol_simple = bench_data.pct_change().std() * np.sqrt(252)
plt.scatter(bench_vol_simple, bench_ret_simple, color='blue', marker='D',
            s=120, label='FTSE MIB (Benchmark)', zorder=5)

plt.title('Efficient Frontier - Mean-Variance Optimization')
plt.xlabel('Risk (Volatility)')
plt.ylabel('Expected Return')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ============================================================
# 5. RISK METRICS — MONTE CARLO PRICE PROJECTION (GBM)
# ============================================================
days = 252
iterations = 10000
initial_investment = 10000

port_log_ret_mean = (log_returns.mean() * best_w).sum() * 252
port_log_vol = np.sqrt(np.dot(best_w.T, np.dot(log_returns.cov() * 252, best_w)))

daily_log_std = port_log_vol / np.sqrt(252)
# Drift corrected with Itô's Calculus (-0.5 * sigma^2)
daily_log_ret = (port_log_ret_mean - 0.5 * port_log_vol**2) / 252

simulated_log_returns = np.random.normal(daily_log_ret, daily_log_std, (days, iterations))
price_series = initial_investment * np.exp(np.cumsum(simulated_log_returns, axis=0))

ending_values = price_series[-1, :]
future_returns = (ending_values - initial_investment) / initial_investment
var_95 = np.percentile(future_returns, 5)

# ============================================================
# 6. CORRELATION HEATMAP (Log-Returns)
# ============================================================
plt.figure(figsize=(8, 6))
sns.heatmap(log_returns.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix (Log-Returns)")
plt.show()

# ============================================================
# 7. HISTORICAL DRAWDOWN (Simple Returns)
# ============================================================
port_simple_ret_hist = (simple_returns * best_w).sum(axis=1)
cumulative_wealth = (1 + port_simple_ret_hist).cumprod()
peak = cumulative_wealth.expanding(min_periods=1).max()
drawdown = (cumulative_wealth / peak) - 1
max_drawdown = drawdown.min()

# ============================================================
# CONSOLE OUTPUT
# ============================================================
print("\n--- PERFORMANCE & RISK SUMMARY ---")
for i in range(len(tickers)):
    print(f"{tickers[i]}: {best_w[i]*100:.2f}%")
print(f"\nVaR 95% (1 Year): {abs(var_95):.2%}")
print(f"Maximum Historical Drawdown: {max_drawdown:.2%}")
print(f"Mean Estimated Final Value: €{ending_values.mean():.2f}")

# ============================================================
# CHART: MONTE CARLO PRICE PATHS (SPAGHETTI CHART)
# ============================================================
plt.figure(figsize=(10, 5))
plt.plot(price_series[:, :100], color='skyblue', alpha=0.3)
plt.plot(price_series.mean(axis=1), color='blue', linewidth=2, label='Mean Path')
plt.axhline(initial_investment * (1 + var_95), color='red', linestyle='--',
            label=f'95% VaR Threshold: {var_95:.2%}')
plt.yscale('log')
plt.title("1-Year Price Simulation (GBM with Itô Correction)")
plt.legend()
plt.show()

# ============================================================
# CHART: HISTORICAL BACKTEST
# ============================================================
plt.figure(figsize=(12, 6))

port_cumulative = (1 + port_simple_ret_hist).cumprod() * initial_investment
bench_simple_ret = bench_data.pct_change().dropna()
bench_simple_ret = bench_simple_ret.reindex(port_simple_ret_hist.index).dropna()
bench_cumulative = (1 + bench_simple_ret).cumprod() * initial_investment

plt.plot(port_cumulative, label=f'Optimal Portfolio (Sharpe: {results[2,max_sharpe_idx]:.2f})',
         color='darkgreen', linewidth=2)
plt.plot(bench_cumulative, label='FTSE MIB (Benchmark)', color='gray',
         linestyle='--', alpha=0.7)
plt.title("Historical Backtest: Optimal Portfolio vs Benchmark (Base €10,000)")
plt.ylabel("Investment Value (€)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ============================================================
# CHART: SIMULATED RETURNS HISTOGRAM
# ============================================================
plt.figure(figsize=(12, 6))
n, bins, patches = plt.hist(future_returns, bins=60, color='skyblue',
                             edgecolor='black', alpha=0.7)
for i in range(len(bins) - 1):
    if bins[i] < 0:
        patches[i].set_facecolor('salmon')

plt.axvline(var_95, color='red', linestyle='dashed', linewidth=2,
            label=f'95% VaR (Critical Threshold): {var_95:.2%}')
plt.axvline(0, color='black', linestyle='-', linewidth=1)
plt.title('Probability Distribution of 1-Year Returns')
plt.xlabel('Return Percentage')
plt.ylabel('Frequency (N. of Simulations)')
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()

# ============================================================
# PERFORMANCE METRICS CALCULATION
# ============================================================
port_ann_ret = np.sum(simple_returns.mean() * best_w) * 252
port_ann_vol = np.sqrt(np.dot(best_w.T, np.dot(simple_returns.cov() * 252, best_w)))
sharpe_ratio = (port_ann_ret - rf) / port_ann_vol

target_daily = 0
downside_diff = np.minimum(port_simple_ret_hist - target_daily, 0)
downside_std = np.sqrt(np.mean(downside_diff**2)) * np.sqrt(252)
sortino_ratio = (port_ann_ret - rf) / downside_std

calmar_ratio = port_ann_ret / abs(max_drawdown)
ulcer_index = np.sqrt(np.mean(drawdown**2))

# ============================================================
# FINAL PERFORMANCE SUMMARY TABLE
# ============================================================
fig, ax_tab = plt.subplots(figsize=(7, 5))
ax_tab.axis('off')

metriche_data = [
    ["Annualized Return",  f"{port_ann_ret:.2%}"],
    ["Annualized Volatility",  f"{port_ann_vol:.2%}"],
    ["Max Drawdown",       f"{max_drawdown:.2%}"],
    ["Sharpe Ratio",       f"{sharpe_ratio:.2f}"],
    ["Sortino Ratio",      f"{sortino_ratio:.2f}"],
    ["Calmar Ratio",       f"{calmar_ratio:.2f}"],
    ["Ulcer Index",        f"{ulcer_index:.4f}"],
]

table = ax_tab.table(
    cellText=metriche_data,
    colLabels=["Performance Indicator", "Value"],
    loc='center',
    cellLoc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2.5)

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor('#D3D3D3')
    if row == 0:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#2C3E50')
    elif row % 2 == 0:
        cell.set_facecolor('#F8F9F9')
    else:
        cell.set_facecolor('white')

ax_tab.set_title("PERFORMANCE & RISK DASHBOARD",
                 fontweight='bold', pad=30, fontsize=14)
plt.tight_layout()
plt.show()