import sys
import subprocess
import pandas as pd
import numpy as np

from datetime import datetime, timedelta

class tools:
  def __init__(self, api_key):
    self.api_key = api_key
    self.install("fugle_marketdata")

  def install(self, package):
    try:    
        subprocess.check_call([sys.executable, "-m", "pip", "show", package])
        print(f"{package} installed already.")
    except subprocess.CalledProcessError:
        print(f"{package} installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"{package} installed successfully.")
  
# ---------------------------------------------------
  def get_history_data(self, symbol):
    from fugle_marketdata import RestClient
    client = RestClient(api_key=self.api_key)
    
    to_date = datetime.today().date()
    from_date = to_date - timedelta(days = 90)
    
    data = client.stock.historical.candles(**{"symbol": symbol, "from": from_date, "to": to_date, "fields": "open,high,low,close,volume,turnover,change"})
    data = data["data"]
    data.sort(key=lambda x: x["date"])
    df = pd.DataFrame(data)

    return df

  def feature_engineering(self, df):
    df["balance"] = df.close - df.open
    df["wma_5"] = self.wma(df.close, 5)
    df["wma_10"] = self.wma(df.close, 10)
    df["wma_20"] = self.wma(df.close, 20)
    df["dif"], df["macd"], df["osc"] = self.macd(df.close)
    df["rsi"] = self.rsi(df.close)
    df["k"], df["d"] = self.kd(df)
    df["bias_6"] = self.bias(df.close, 6)
    df["bias_12"] = self.bias(df.close, 12)
    df["bias_24"] = self.bias(df.close, 24)
    df["upper_band"], df["lower_band"]= self.bollinger_bands(df.close)

    return df

  def normalize(self, df):
    max_val = df[['open', 'high', 'low', 'close']].max().max()
    min_val = df[['open', 'high', 'low', 'close']].min().min()

    for col in ['open', 'high', 'low', 'close', 'volume', 'wma_5', 'wma_10','wma_20', 'upper_band', 'lower_band']:
      df[f"scaled_{col}"] = np.round((df[[col]] - min_val) / (max_val - min_val), 6)

    df[f"scaled_volume"] =  np.round((df.volume - df.volume.min()) / (df.volume.max() - df.volume.min()), 6)

    for col in ['rsi', 'k', 'd', 'bias_6', 'bias_12', 'bias_24']:
      df[f"scaled_{col}"] = np.round((df[[col]] / 100), 6)

    df["range"] = np.round((df.close - df.open) / df.open, 6)
    df.dropna(inplace=True)
    df.reset_index(drop = True, inplace = True)

    return df

  def assemble(self, df):
    scaled_columns = [col for col in df.columns if col.startswith('scaled_')]
    scaled_columns.append("osc")

    xy_df = pd.DataFrame()
    xy_df["X"] = df[scaled_columns].values.tolist()
    xy_df["y"] = df.apply(lambda row: row.range, axis = 1)
    features = len(xy_df.X.values[0])

    return xy_df, features

  def windowed(self, df, window_size, test_size):
    win_df = pd.DataFrame()
    for i in range(window_size):
      win_df[f"x_{i}"] = df.X.shift(i)

    win_df["y"] = df["y"]
    # win_df["y"] = df["y"].shift(-1)
    # win_df.y[-1:].fillna(0, inplace=True)
    win_df.dropna(inplace=True)
    win_df.reset_index(drop = True, inplace = True)
    win_df = win_df.tail(test_size)

    return win_df

  def process_data(self, data, window_size, features):
    stacked_data = np.concatenate([np.stack(data[col].values) for col in data.columns], axis=1)
    reshaped_data = stacked_data.reshape(stacked_data.shape[0], window_size, features)
    
    return reshaped_data
 
  # def test_data(y_df, test_size):
  #   X_test = df.tail(test_size)
  #   y_test = df.tail(test_size)  
  # model20140817.keras

# -----------indicators------------------------------------------------
  def sma(self, data, window):
    return data.rolling(window).mean()
  
  def ema(self, data, span):
    return data.ewm(span=span, adjust=False).mean()  

  def wma(self, data, window):
    weights = np.arange(1, window + 1)
    wma = data.rolling(window).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
    return wma

  def macd(self, data, short_window=12, long_window=26, signal_window=9):
    short_ema = self.ema(data, short_window)
    long_ema = self.ema(data, long_window)
    macd_line = short_ema - long_ema
    signal_line = self.ema(macd_line, signal_window)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

  def rsi(self, data, window = 14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

  def kd(self, data, period=14):
    low = data.low.rolling(window=period).min()
    high = data.high.rolling(window=period).max()
    k = 100 * ((data.close - low) / (high - low))
    d = k.rolling(window=3).mean()
    return k, d
        
  def bias(self, data, window):
    ma = self.sma(data, window)
    return ((data - ma) / ma) * 100

  def bollinger_bands(self, data, window=20, num_std=2):
    mean = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    upper_band = mean + (std * num_std)
    lower_band = mean - (std * num_std)
    return upper_band, lower_band
