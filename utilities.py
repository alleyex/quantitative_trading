import subprocess
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

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

  def initialize_plot(self):
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    line_loss, = ax.plot([], [], label="Loss")
    ax.grid(True)
    ax.legend()
    return fig, ax, line_loss

  def on_epoch_end(self, epoch, logs, losses, line_loss, ax, fig):
    losses.append(logs["loss"])
    line_loss.set_xdata(range(epoch + 1))
    line_loss.set_ydata(losses)
    ax.relim()
    ax.autoscale_view()
    display(fig)
    clear_output(wait = True)

  def create_plot_callback(self):
    fig, ax, line_loss = self.initialize_plot()
    losses = []
    callback = lambda epoch, logs: self.on_epoch_end(epoch, logs, losses, line_loss, ax, fig)
    
    return tf.keras.callbacks.LambdaCallback(on_epoch_end = callback)

        
