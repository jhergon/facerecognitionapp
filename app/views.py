import os
import cv2
from app.face_recognition import faceRecognitionPipeline
from flask import render_template, request, url_for
import matplotlib.image as matimg

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quandl as ql
import matplotlib
matplotlib.use('agg')  # Usar el backend 'agg'
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

UPLOAD_FOLDER = 'static/upload'

def index():
    return render_template('index.html')


def app():
    return render_template('app.html')

def calculator():
    return render_template('calculator.html')

def DOIschain():
    return render_template('DOIschain.html')


def genderapp():
    if request.method == 'POST':
        f = request.files['image_name']
        filename = f.filename
        # save our image in upload folder
        path = os.path.join(UPLOAD_FOLDER,filename)
        f.save(path) # save image into upload folder
        # get predictions
        pred_image, predictions = faceRecognitionPipeline(path)
        pred_filename = 'prediction_image.jpg'
        cv2.imwrite(f'./static/predict/{pred_filename}',pred_image)
        
        # generate report
        report = []
        for i , obj in enumerate(predictions):
            gray_image = obj['roi'] # grayscale image (array)
            eigen_image = obj['eig_img'].reshape(100,100) # eigen image (array)
            gender_name = obj['prediction_name'] # name 
            score = round(obj['score']*100,2) # probability score
            
            # save grayscale and eigne in predict folder
            gray_image_name = f'roi_{i}.jpg'
            eig_image_name = f'eigen_{i}.jpg'
            matimg.imsave(f'./static/predict/{gray_image_name}',gray_image,cmap='gray')
            matimg.imsave(f'./static/predict/{eig_image_name}',eigen_image,cmap='gray')
            
            # save report 
            report.append([gray_image_name,
                           eig_image_name,
                           gender_name,
                           score])
            
        
        return render_template('gender.html',fileupload=True,report=report) # POST REQUEST
            
    
    
    return render_template('gender.html',fileupload=False) # GET REQUEST

def tradingapp():

    # Get data
   # dataset = ql.get("CHRIS/CME_ES1", start_date="", end_date="", api_key='hiqkVfCKR65jR9y7yvoM')
    dataset = ql.get("CURRFX/USDEUR", start_date="", end_date="", api_key='hiqkVfCKR65jR9y7yvoM')
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
    plot_file = 'static/plot.PNG'
    plt.savefig(plot_file)
    plt.close()

    #return render_template('trading.html', plot_file=plot_file)
    # Generar la URL relativa para la imagen del gráfico
    plot_file_relative = url_for('static', filename='plot.png')

    return render_template('trading.html', plot_file=plot_file_relative)



def tradingapp2():
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

    # Prepare data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data = scaler.fit_transform(dataset.iloc[:, 4:].values)
    n = data.shape[0]
    train_end = int(np.floor(0.8 * n))
    X_train, Y_train = data[:train_end, :-1], data[:train_end, -1]
    X_test, Y_test = data[train_end:, :-1], data[train_end:, -1]
    n_features = X_train.shape[1]

    # Neural network parameters
    n_neurons = [512, 256, 128]

    # Build neural network
    X = tf.keras.layers.Input(shape=(n_features,))
    layer = X
    for neurons in n_neurons:
        layer = tf.keras.layers.Dense(neurons, activation='relu')(layer)

    out = tf.keras.layers.Dense(1)(layer)
    model = tf.keras.Model(inputs=X, outputs=out)
    model.compile(optimizer='adam', loss='mse')

    # Train neural network
    batch_size = 200
    epochs = 10
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs)

    # Predictions
    pred = model.predict(X_test)

    # Threshold predictions
    y_pred = (pred > 0.5).astype(np.float32)

    # Trading strategy
    dataset['y_pred'] = np.nan
    dataset.iloc[-len(y_pred):, -1:] = y_pred
    trade_dataset = dataset.dropna()

    trade_dataset['Return_Tomorrow'] = np.log(trade_dataset['Last'].shift(-1) / trade_dataset['Last'])
    trade_dataset['Strategy'] = np.where(trade_dataset['y_pred'] == 1, -trade_dataset['Return_Tomorrow'], trade_dataset['Return_Tomorrow'])
    trade_dataset['Market_Accumulated'] = trade_dataset['Return_Tomorrow'].cumsum()
    trade_dataset['Strategy_Accumulated'] = trade_dataset['Strategy'].cumsum()

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(trade_dataset['Market_Accumulated'], label='Market_Accumulated')
    plt.plot(trade_dataset['Strategy_Accumulated'], label='Strategy_Accumulated')
    plt.legend()

    # Save the plot to a file
    plot_file = 'static/plot.png'
    plt.savefig(plot_file)
    plt.close()

     #return render_template('trading.html', plot_file=plot_file)
    # Generar la URL relativa para la imagen del gráfico
    plot_file_relative = url_for('static', filename='plot.png')

    return render_template('trading.html', plot_file=plot_file_relative)




