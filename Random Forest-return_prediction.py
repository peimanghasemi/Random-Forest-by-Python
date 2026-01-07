#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('pip', 'install numpy pandas yfinance matplotlib scikit-learn')


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

tickers = {
    "Asset 1": "aapl.us",
    "Asset 2": "msft.us",
    "Asset 3": "amzn.us",
    "Asset 4": "googl.us",
    "Asset 5": "tsla.us",
    "Asset 6": "nvda.us",
    "Asset 7": "meta.us",
    "Asset 8": "jpm.us",
    "Asset 9": "jnj.us",
    "Asset 10": "xom.us",
}

start_date = "2018-03-07"
end_date = "2022-09-07"

transaction_cost = 0.001
risk_free_rate = 0.02
trading_days = 252

def load_stooq(symbol):
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    df = pd.read_csv(url)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    df = df.loc[(df.index >= start_date) & (df.index <= end_date)]
    if "Close" not in df.columns:
        return pd.DataFrame()
    return df[["Close"]].dropna()

def max_drawdown(cum_wealth):
    peak = np.maximum.accumulate(cum_wealth)
    dd = (cum_wealth / peak) - 1.0
    return dd.min()

results = []
plotted = False

for asset_name, symbol in tickers.items():
    df = load_stooq(symbol)
    if df.empty or len(df) < 150:
        print(f"Skipping {symbol} (no/low data)")
        continue

    df["Return"] = df["Close"].pct_change()
    df.dropna(inplace=True)

    df["Lag_1"] = df["Return"].shift(1)
    df["Lag_5"] = df["Return"].shift(5)
    df["Rolling_Mean_10"] = df["Return"].rolling(10).mean()
    df["Rolling_Std_10"] = df["Return"].rolling(10).std()
    df.dropna(inplace=True)

    cut = int(len(df) * 0.8)
    train = df.iloc[:cut]
    test = df.iloc[cut:]

    X_train = train[["Lag_1", "Lag_5", "Rolling_Mean_10", "Rolling_Std_10"]]
    y_train = train["Return"]
    X_test = test[["Lag_1", "Lag_5", "Rolling_Mean_10", "Rolling_Std_10"]]
    y_test = test["Return"].values

    if len(X_train) == 0 or len(X_test) == 0:
        print(f"Skipping {symbol} (empty train/test)")
        continue

    model = RandomForestRegressor(
        n_estimators=60,
        max_depth=10,
        random_state=42,
        bootstrap=True
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred, squared=False)
    r2 = r2_score(y_test, pred)

    pos = (pred > 0).astype(int)
    strat = pos * y_test
    trades = np.abs(np.diff(pos, prepend=0))
    strat = strat - trades * transaction_cost

    ann_ret = strat.mean() * trading_days
    ann_vol = strat.std() * np.sqrt(trading_days)

    sharpe = np.nan if ann_vol == 0 else (ann_ret - risk_free_rate) / ann_vol

    downside = strat[strat < 0]
    dvol = downside.std() * np.sqrt(trading_days) if len(downside) > 0 else 0
    sortino = np.nan if dvol == 0 else (ann_ret - risk_free_rate) / dvol

    wealth = (1 + strat).cumprod()
    mdd = max_drawdown(wealth)
    calmar = np.nan if mdd == 0 else ann_ret / abs(mdd)

    results.append({
        "Asset": asset_name,
        "Symbol": symbol,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "AnnualReturn_%": ann_ret * 100,
        "AnnualVol_%": ann_vol * 100,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "MaxDrawdown_%": mdd * 100
    })

    if not plotted:
        plotted = True
        dates = test.index

        plt.figure(figsize=(12, 5))
        plt.plot(dates, y_test, label="Actual", linewidth=1)
        plt.plot(dates, pred, label="Predicted", linestyle="dashed", linewidth=1)

        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
        plt.xticks(rotation=45)

        plt.title(f"Actual vs Predicted Returns (sample: {symbol})")
        plt.xlabel("Date")
        plt.ylabel("Daily Return")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

if len(results) == 0:
    print("No assets processed. Possibly network blocked.")
else:
    res = pd.DataFrame(results)

    print("\nModel performance (per asset)")
    print(res[["Asset", "Symbol", "MAE", "RMSE", "R2"]].round(6).to_string(index=False))

    print("\nSimple strategy performance (based on sign(pred))")
    print(res[["Asset", "Symbol", "AnnualReturn_%", "AnnualVol_%", "Sharpe", "Sortino", "Calmar", "MaxDrawdown_%"]]
          .round(4).to_string(index=False))


# In[ ]:




