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
    return fig, ax, line_loss

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

        

        
