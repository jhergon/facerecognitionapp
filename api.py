from flask import Flask, render_template
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quandl as ql

app = Flask(__name__)

@app.route('/')
def home():
    # Get data
    dataset = ql.get("CHRIS/CME_ES1", start_date="", end_date="", api_key='hiqkVfCKR65jR9y7yvoM')
    dataset = dataset.dropna()
    dataset = dataset[['Open', 'High', 'Low', 'Last']]
    dataset['H-L'] = dataset['Last'] - dataset['Open']
    dataset['O-C'] = dataset['High'] - dataset['Low']
    dataset['10day MA'] = dataset['Last'].shift(1).rolling(window=10).mean()
    dataset['20day MA'] = dataset['Last'].shift(1).rolling(window=20).mean()
    dataset['55day MA'] = dataset['Last'].shift(1).rolling(window=55).mean()
    dataset['stdev'] = dataset['Last'].rolling(window=55).std()
    dataset['Price_Rise'] = np.where(dataset['Last'].shift(-1) > dataset['Last'], 1, 0)
    dataset = dataset.dropna()

    # Trading strategy
    dataset['Return_Tomorrow'] = np.log(dataset['Last'].shift(-1) / dataset['Last'])
    dataset['Market_Accumulated'] = dataset['Return_Tomorrow'].cumsum()

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(dataset['Market_Accumulated'], label='Market_Accumulated')
    plt.legend()

    # Save the plot to a file
    plot_file = 'static/plot.png'
    plt.savefig(plot_file)
    plt.close()

    return render_template('trading.html', plot_file=plot_file)
    


if __name__ == '__main__':
    app.run(debug=True)
