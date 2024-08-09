import subprocess
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from IPython.display import display, clear_output

class initializing:
      
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
   

  def download_data(self, symbol):
    from fugle_marketdata import RestClient
    client = RestClient(api_key=self.api_key)

    data = []
    for year in range(2014, 2025):
      from_date = f'{year}-01-01'
      to_date = f'{year}-12-31'
      yearly_data = client.stock.historical.candles(**{"symbol": symbol, "from": from_date, "to": to_date, "fields": "open,high,low,close,volume,change"})
      yearly_data = yearly_data['data']
      yearly_data.sort(key=lambda x: x['date'])
      data.extend(yearly_data)

    return data

  def get_data(self, symbol):
    file_name = symbol + ".csv"

    if os.path.isfile(file_name):
      df = pd.read_csv(file_name)
      print(f"{file_name}: exist! \n")
    else:
      data = self.download_data(symbol)
      df = pd.DataFrame(data)
      df.to_csv(file_name, index=False)
      print(f"{file_name}: download! \n")

    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by="date", inplace = True)

    return df

# ----------------------------------------------------------------------
      
  def initialize_plot(self, metrics):
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")        
    lines_dict = {metric: ax.plot([], [], label=metric)[0] for metric in metrics}
    losses_dict = {metric: [] for metric in metrics}
        
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

# -----------------------------------------------------------------------
  def show_naive(self, symbol, days = 20):
    df = self.get_data(symbol)
    df["range"] = np.round((df.close - df.open) / df.open, 4)
    df["y_hat"] = df.range
    df["y"] = df.range.shift(-1)

    df.dropna(inplace=True)

    df = df[["y_hat", "y"]].tail(days)
    self.show_compare_data(df)

  def show_compare_data(self, df):
    df["win"] = (df.y_hat > 0) & (df.y > 0)
    df["win"] = df.win.astype(int)
    
    df.loc[df["y_hat"] <= 0, "win"] = 0
    df.loc[(df["y_hat"] > 0) & (df["y"] <= 0), "win"] = -1


    counts = df.win.value_counts().to_string()
    print(f"Win Value Counts: \n {counts}")
        
    balance = df.loc[df["win"] != 0, "y"].sum() *100
    print(f"\nBalance: {balance:.2f} %\n")
        
    print("Comparison DataFrame:\n\n", df)


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

# ---------------------------------------------------------------
  def plot_predict(self, model, df, X_test, test_size):
    results = model.predict(X_test)

    y_hat = results.flatten()
    y = df.tail(test_size + 1).close.values
    y = y[:-1]

    fig, ax1 = plt.subplots(figsize = (12, 6))

    ax2 = ax1.twinx()
    ax1.plot(y, label = "Actual", color = "blue")
    ax2.plot(y_hat, label = "Predicted", color = "red")

    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax1.grid(True)

    plt.legend()
    plt.show()







        

        
