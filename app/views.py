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

UPLOAD_FOLDER = 'static/upload'

def index():
    return render_template('index.html')


def app():
    return render_template('app.html')

def calculator():
    return render_template('calculator.html')


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
    plot_file = 'static/plot.PNG'
    plt.savefig(plot_file)
    plt.close()

    #return render_template('trading.html', plot_file=plot_file)
    # Generar la URL relativa para la imagen del gráfico
    plot_file_relative = url_for('static', filename='plot.png')

    return render_template('trading.html', plot_file=plot_file_relative)
    


