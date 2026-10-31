import pandas as pd
import numpy as np

def miss_sniper(df):
    df = df.copy()

    df["ema50"] = df["close"].ewm(span=50).mean()
    df["ema200"] = df["close"].ewm(span=200).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())

    true_range = pd.concat(
        [high_low, high_close, low_close], axis=1
    ).max(axis=1)

    df["atr"] = true_range.ewm(
        alpha=1/14, adjust=False
    ).mean()

    df["buy"] = (
        (df["ema50"] > df["ema200"]) &
        (df["close"] > df["ema50"]) &
        (df["rsi"] > 50) &
        (df["rsi"].shift(1) <= 50)
    )

    df["sell"] = (
        (df["ema50"] < df["ema200"]) &
        (df["close"] < df["ema50"]) &
        (df["rsi"] < 50) &
        (df["rsi"].shift(1) >= 50)
    )

    df["stop_loss_distance"] = df["atr"] * 1.5
    df["take_profit_distance"] = df["atr"] * 3

    return df
