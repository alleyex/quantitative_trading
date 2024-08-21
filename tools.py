import os
import sys
import time
import subprocess
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from IPython.display import display, clear_output
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
  def get_history_data(self, test_symbol, symbols, window_size, test_size):
    if test_symbol in symbols:
      symbols.remove(test_symbol)
      symbols.append(test_symbol)

    total_df = pd.DataFrame()
    features = None

    for symbol in symbols:
      df = self.check_file(symbol)
      df = self.feature_engineering(df)
      df = self.normalize(df)
      df, features = self.assemble(df)
      df = self.windowed(df, window_size)
      if test_symbol != symbol:
        df = df[: -1]
      print("------------------------------------\n")
      total_df = pd.concat([total_df, df], ignore_index=True)      

    return total_df, features

  def check_file(self, symbol):
    
    file_name = symbol + ".csv"

    if os.path.isfile(file_name):
      df = pd.read_csv(file_name)
      print(f"{file_name}: exist!     {df.shape}")
    else:
      data = self.get_data(symbol)
      df = pd.DataFrame(data)
      df.to_csv(file_name, index=False)
      print(f"{file_name}: download!   {df.shape}")
      time.sleep(20)

    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by="date", inplace = True)

    return df

  def get_data(self, symbol):
    from fugle_marketdata import RestClient
    client = RestClient(api_key=self.api_key)

    this_year = datetime.now().year
    
    data = []
    for year in range(2010, this_year + 1):
      from_date = f"{year}-01-01"
      to_date = f"{year}-12-31"
      yearly_data = client.stock.historical.candles(**{"symbol": symbol, "from": from_date, "to": to_date, "fields": "open,high,low,close,volume,change"})
      
      if yearly_data.get("statusCode"):
        print(f"\n[{symbol} in {year}] \nStatus Code : {yearly_data['statusCode']} Message : {yearly_data['message']}")
      else:
        yearly_data = yearly_data['data']
        yearly_data.sort(key=lambda x: x['date'])
        data.extend(yearly_data)
    return data

  # for predict only
  def get_test_data(self, symbol, test_size):
    df = self.check_file(symbol)
    df = df.tail(test_size)

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
   
    print(f"Feature Engineering: {df.shape}")
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

    print(f"normalizing :        {df.shape}")

    return df

  def assemble(self, df):
    scaled_columns = [col for col in df.columns if col.startswith('scaled_')]
    scaled_columns.append("osc")

    xy_df = pd.DataFrame()
    xy_df["X"] = df[scaled_columns].values.tolist()
    xy_df["y"] = df.apply(lambda row: row.range, axis = 1)
    features = len(xy_df.X.values[0])

    print(f"Features & Lables :  {xy_df.shape}    Number of Features:  {features}")

    return xy_df, features

  def windowed(self, df, window_size):
    win_df = pd.DataFrame()
    for i in range(window_size):
      win_df[f"x_{i}"] = df.X.shift(i)

    win_df["y"] = df["y"].shift(-1)
    win_df.y[-1:].fillna(0, inplace=True)
    win_df.dropna(inplace=True)
    win_df.reset_index(drop = True, inplace = True)


    print(f"windowed Size = {window_size} :  {win_df.shape}")

    return win_df

  def process_data(self, data, window_size, features):
    stacked_data = np.concatenate([np.stack(data[col].values) for col in data.columns], axis=1)
    reshaped_data = stacked_data.reshape(stacked_data.shape[0], window_size, features)
    
    print(f"reshaped Data :         {reshaped_data.shape}")
    return reshaped_data
 
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

# ----------------------------------------------------------------------
      
  def initialize_plot(self, metrics):
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")        
    lines_dict = {metric: ax.plot([], [], label=metric)[0] for metric in metrics}
    losses_dict = {metric: [] for metric in metrics}      
    
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))    
    ax.grid(True)
    ax.legend()
    return fig, ax, lines_dict, losses_dict

  def on_epoch_end(self, epoch, logs, losses_dict, lines_dict, ax, fig):
    for key in losses_dict.keys():
      losses_dict[key].append(logs[key])
      lines_dict[key].set_xdata(range(epoch + 1))
      lines_dict[key].set_ydata(losses_dict[key])
        
    ax.relim()
    ax.autoscale_view()
    ax.legend()
    display(fig)
    clear_output(wait = True)

  def create_plot_callback(self, metrics = ["loss"]):
    fig, ax, lines_dict, losses_dict = self.initialize_plot(metrics)   
        
    callback = lambda epoch, logs: self.on_epoch_end(epoch, logs, losses_dict, lines_dict, ax, fig)
    
    return tf.keras.callbacks.LambdaCallback(on_epoch_end = callback)


  def plot_predict(self, y_hat, y):  
    fig, ax1 = plt.subplots(figsize = (12, 6))

    ax2 = ax1.twinx()
    ax1.plot(y, label = "Actual", color = "blue")
    ax2.plot(y_hat, label = "Predicted", color = "red")

    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax1.grid(True)

    plt.legend()
    plt.show()


# -----------------------------------------------------------------------
  def show_naive(self, df):
 
    df_range = np.round((df.close - df.open) / df.open, 4)
   
    evalu = pd.DataFrame()
    evalu["y_hat"] = df_range
    evalu["y"] = df_range.shift(-1)

    evalu.dropna(inplace = True)  
    evalu.reset_index(inplace = True, drop = True)  
    
    self.show_evaluation(evalu)

  def show_evaluation(self, df):

    df["win"] = (df.y_hat > 0) & (df.y > 0)
    df["win"] = df.win.astype(int)
    
    df.loc[df["y_hat"] <= 0, "win"] = 0
    df.loc[(df["y_hat"] > 0) & (df["y"] <= 0), "win"] = -1


    counts = df.win.value_counts().to_string()
    print(f"Win Value Counts: \n {counts}")
        
    balance = df.loc[df["win"] != 0, "y"].sum() *100
    print(f"\nBalance: {balance:.2f} %\n")
        
    print("Comparison DataFrame:\n\n", df)
