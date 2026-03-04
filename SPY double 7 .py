#!/usr/bin/env python3
"""
================================================================================
DOUBLE 7 TRADING STRATEGY BACKTEST
Based on: https://github.com/wumbrath/double7
Adapted to use the reporting structure from 'one_percent_per_week_tqqq.py'
================================================================================

Strategy Logic (from original notebook):
- Uses S&P 500 (^GSPC) data
- Entries: when yesterday was a 7‑day low AND yesterday’s close > 200‑day MA
- Exits depend on the variant:
    * Strat: exit on today's 7‑day high
    * Strat_5ma_stop: exit when close > 5‑day MA
    * Strat_200ma_stop: exit on 7‑day high OR close < 200‑day MA
    * Strat_0d_hold: exit same day
    * Strat_trail: trailing stop based on daily lows
    * BuyHold: benchmark buy‑and‑hold
- Trades are entered at the open on the signal day, exited at the close (or stop)
- Returns and equity curves are calculated for each system

REQUIREMENTS:
pip install yfinance quantstats pandas numpy matplotlib
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
import quantstats as qs
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =================================================================================
# CONFIGURATION
# =================================================================================
SYMBOL = "^GSPC"                     # S&P 500
BENCHMARK = "^GSPC"                  # benchmark (same, or use SPY)
START_DATE = "1992-01-01"
END_DATE = datetime.now().strftime('%Y-%m-%d')
STARTING_BALANCE = 10000
SLOW_MA = 200
FAST_MA = 5
PERIODS = [7]                         # can be extended to range(2,11)
SYSTEMS = ["Strat", "Strat_5ma_stop", "BuyHold"]  # add others if needed
LEVERAGE_FACTOR = 1.0                  # no leverage (kept for compatibility)

# =================================================================================
# DATA LOADING (with caching, same as reference)
# =================================================================================
def load_data(symbol, start, end):
    """Load and cache price data, handling yfinance changes."""
    cache_dir = "data_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{symbol.replace('^','')}.parquet")

    if os.path.exists(cache_file):
        print(f"  Loading {symbol} from cache...")
        df = pd.read_parquet(cache_file)
    else:
        print(f"  Downloading {symbol}...")
        df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.to_parquet(cache_file)

    # Ensure we have OHLC data
    required = ['Open', 'High', 'Low', 'Close']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Filter to exact date range
    df = df[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))]
    return df

# =================================================================================
# STRATEGY CALCULATIONS (from original notebook)
# =================================================================================
def calculate_inputs(df, per):
    """Add indicators: moving averages, 7‑day low/high."""
    df = df.copy()
    df['Fast_MA'] = df['Close'].rolling(window=FAST_MA).mean()
    df['Slow_MA'] = df['Close'].rolling(window=SLOW_MA).mean()
    df['Low_7'] = df['Close'] == df['Close'].rolling(window=per).min()
    df['High_7'] = df['Close'] == df['Close'].rolling(window=per).max()
    return df

def generate_signals(df, system):
    """Create entry/exit signal columns."""
    df = df.copy()
    if system == "BuyHold":
        df[f"{system}_Signal"] = True
    else:
        # Entry: yesterday was 7‑day low AND yesterday's close > slow MA
        df[f"{system}_Signal"] = (
            df['Low_7'].shift(1) & (df['Close'].shift(1) > df['Slow_MA'].shift(1))
        )
        if system == "Strat":
            df[f"{system}_Exit"] = df['High_7']
        elif system == "Strat_200ma_stop":
            df[f"{system}_Exit"] = df['High_7'] | (df['Close'] < df['Slow_MA'])
        elif system == "Strat_5ma_stop":
            df[f"{system}_Exit"] = df['Close'] > df['Fast_MA']
        elif system == "Strat_0d_hold":
            df[f"{system}_Exit"] = True
        elif system == "Strat_trail":
            df[f"{system}_Exit"] = False   # handled separately
    return df

def generate_trades(df, system):
    """Simulate trades and build equity curve."""
    df = df.copy()
    if system == "BuyHold":
        df[f"{system}_Ret"] = df['Close'] / df['Close'].shift(1)
        df[f"{system}_Ret"].iat[0] = 1
        df[f"{system}_Bal"] = STARTING_BALANCE * df[f"{system}_Ret"].cumprod()
        df[f"{system}_In_Market"] = True
        return df

    trades_list = []
    trade_open = False
    open_change = {}
    stop_loss = 0
    entry_price = 0
    entry_date = None

    for i, (idx, row) in enumerate(df.iterrows()):
        if not trade_open:
            if row[f"{system}_Signal"]:
                entry_date = idx
                entry_price = row['Open']
                trade_open = True
        else:
            # Check normal exit signal
            if row[f"{system}_Exit"]:
                exit_date = idx
                exit_price = row['Close']
                trade_open = False
                trades_list.append([entry_date, entry_price, exit_date, exit_price, True])
            else:
                open_change[idx] = row['Low'] / entry_price

                # Trailing stop logic
                if system == "Strat_trail":
                    days_held = np.busday_count(entry_date.date(), idx.date())
                    if days_held == 0:
                        if row['Close'] <= entry_price:
                            exit_date = idx
                            exit_price = row['Close']
                            trade_open = False
                            trades_list.append([entry_date, entry_price, exit_date, exit_price, True])
                        else:
                            stop_loss = row['Low']
                    else:
                        if stop_loss:
                            if row['Open'] <= stop_loss:
                                exit_date = idx
                                exit_price = row['Open']
                                trade_open = False
                                stop_loss = 0
                                trades_list.append([entry_date, entry_price, exit_date, exit_price, True])
                            elif row['Low'] <= stop_loss:
                                exit_date = idx
                                exit_price = stop_loss
                                trade_open = False
                                stop_loss = 0
                                trades_list.append([entry_date, entry_price, exit_date, exit_price, True])
                            else:
                                stop_loss = row['Low']

    # Build trade DataFrame
    if trades_list:
        trades = pd.DataFrame(trades_list,
                              columns=['Entry_Date', 'Entry_Price', 'Exit_Date', 'Exit_Price', 'Sys_Trade'])
        trades[f"{system}_Return"] = trades['Exit_Price'] / trades['Entry_Price']
        # Duration in business days
        durations = []
        for _, t in trades.iterrows():
            d1 = t['Entry_Date']
            d2 = t['Exit_Date']
            durations.append(np.busday_count(d1.date(), d2.date()) + 1)
        trades[f"{system}_Duration"] = durations

        # Create returns series indexed by exit date
        returns = pd.DataFrame(index=trades['Exit_Date'])
        returns[f"{system}_Ret"] = trades[f"{system}_Return"].values
        returns[f"{system}_Trade"] = True
        returns[f"{system}_Duration"] = trades[f"{system}_Duration"].values

        # Entry prices indexed by entry date
        entries = pd.DataFrame(index=trades['Entry_Date'])
        entries[f"{system}_Entry_Price"] = trades['Entry_Price'].values

        # Merge
        df = pd.concat([df, returns, entries], axis=1)

    # Fill missing
    df[f"{system}_Ret"] = df.get(f"{system}_Ret", pd.Series(1, index=df.index)).fillna(1)
    df[f"{system}_Trade"] = df.get(f"{system}_Trade", pd.Series(False, index=df.index)).fillna(False)
    df[f"{system}_Bal"] = STARTING_BALANCE * df[f"{system}_Ret"].cumprod()

    # Running balance (accounts for intra‑trade drawdown)
    change_series = pd.Series(open_change, name=f"{system}_Change")
    df = pd.concat([df, change_series], axis=1)
    df[f"{system}_Change"] = df[f"{system}_Change"].fillna(1)
    df[f"{system}_Running_Bal"] = df[f"{system}_Bal"] * df[f"{system}_Change"]

    # Mark in‑market days
    df[f"{system}_In_Market"] = False
    for idx, is_trade in df[f"{system}_Trade"].items():
        if is_trade:
            dur = df.loc[idx, f"{system}_Duration"]
            pos = df.index.get_loc(idx)
            for back in range(int(dur)):
                if pos - back >= 0:
                    df.iloc[pos - back, df.columns.get_loc(f"{system}_In_Market")] = True

    return df

def run_backtest(prices, period):
    """Run all systems for a given period and return dict of returns and metrics."""
    print(f"\n[ENGINE] Running Double7 backtest for period={period}...")
    df = prices.copy()
    for sys in SYSTEMS:
        df = calculate_inputs(df, period)
        df = generate_signals(df, sys)
        df = generate_trades(df, sys)

    # Collect returns and equity for each system
    returns_dict = {}
    equity_dict = {}
    for sys in SYSTEMS:
        ret_col = f"{sys}_Ret"
        bal_col = f"{sys}_Bal"
        if ret_col in df.columns:
            returns_dict[sys] = df[ret_col].dropna()
        if bal_col in df.columns:
            equity_dict[sys] = df[bal_col]

    # Compute summary metrics (for the first system as example, but we'll report all later)
    metrics = {}
    for sys in SYSTEMS:
        if sys in equity_dict:
            eq = equity_dict[sys]
            total_ret = (eq.iloc[-1] / eq.iloc[0]) - 1
            years = (eq.index[-1] - eq.index[0]).days / 365.25
            cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if years > 0 else 0
            metrics[sys] = {
                'final_value': eq.iloc[-1],
                'total_return': total_ret,
                'cagr': cagr
            }
    return returns_dict, equity_dict, metrics, df

# =================================================================================
# MAIN EXECUTION
# =================================================================================
def main():
    print("="*80)
    print("DOUBLE 7 TRADING STRATEGY BACKTEST")
    print("="*80)

    print(f"\n[DATA] Loading {SYMBOL}...")
    prices = load_data(SYMBOL, START_DATE, END_DATE)
    benchmark_prices = load_data(BENCHMARK, START_DATE, END_DATE) if BENCHMARK != SYMBOL else prices

    print(f"[DATA] Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"[DATA] Trading days: {len(prices)}")

    for period in PERIODS:
        print(f"\n{'='*60}")
        print(f"PERIOD = {period} DAYS")
        print('='*60)

        returns_dict, equity_dict, metrics, full_df = run_backtest(prices, period)

        # Plot equity curves
        plt.figure(figsize=(12,6))
        for sys, eq in equity_dict.items():
            plt.plot(eq.index, eq.values, label=sys)
        plt.title(f"Equity Curves (period={period}) - {SYMBOL}")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value ($)")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Print metrics
        print("\n[METRICS] Performance by System:")
        for sys, m in metrics.items():
            print(f"\n{sys}:")
            print(f"  Final Value: ${m['final_value']:,.2f}")
            print(f"  Total Return: {m['total_return']*100:.2f}%")
            print(f"  CAGR: {m['cagr']*100:.2f}%")

        # Generate QuantStats report for each system (optional, here we pick the first non‑BuyHold)
        print("\n[REPORT] Generating QuantStats Analysis...")
        output_dir = "reports"
        os.makedirs(output_dir, exist_ok=True)

        for sys in SYSTEMS:
            if sys in returns_dict and sys != "BuyHold":   # avoid duplicate benchmark
                ret_series = returns_dict[sys]
                # Align benchmark returns
                bench_ret = benchmark_prices['Close'].pct_change().dropna()
                common_idx = ret_series.index.intersection(bench_ret.index)
                if len(common_idx) > 0:
                    ret_aligned = ret_series.loc[common_idx]
                    bench_aligned = bench_ret.loc[common_idx]
                    out_file = os.path.join(output_dir, f"Double7_{sys}_p{period}.html")
                    qs.reports.html(ret_aligned, benchmark=bench_aligned,
                                    output=out_file,
                                    title=f"Double7 {sys} (period={period})")
                    print(f"  Saved: {out_file}")

        # Save trades (if any) for one system as example
        for sys in SYSTEMS:
            if sys != "BuyHold" and f"{sys}_Trade" in full_df.columns:
                trades = full_df[full_df[f"{sys}_Trade"] == True]
                if not trades.empty:
                    trades_file = os.path.join(output_dir, f"trades_{sys}_p{period}.csv")
                    trades.to_csv(trades_file)
                    print(f"  Trades saved: {trades_file}")
                break

    print("\n" + "="*80)
    print("NOTE: The Double7 strategy enters on 7‑day lows and exits on 7‑day highs")
    print("or other conditions depending on the variant. See original notebook.")
    print("="*80)

if __name__ == "__main__":
    main()