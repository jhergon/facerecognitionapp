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

import io
import base64
import requests

UPLOAD_FOLDER = 'static/upload'
# Your GitHub token (replace with your own token)


def index():
    return render_template('index.html')


def app():
    return render_template('app.html')

def calculator():
    return render_template('calculator.html')

def rif():
    return render_template('rif.htm')

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



############
#toco cambiar la quanl porque cambio de empresa
#https://github.com/matplotlib/mplfinance/issues/129
#https://github.com/Nasdaq/data-link-python/#local-api-key-environment-variable
#https://github.com/quandl/quandl-python?tab=readme-ov-file
#https://docs.data.nasdaq.com/docs/python-tables
#endpoint
#https://data.nasdaq.com/api/v3/datatables/QDL/FON/metadata?api_key=hiqkVfCKR65jR9y7yvoM
#https://docs.data.nasdaq.com/docs/in-depth-usage-1
#https://www.quantconnect.com/docs/v2/writing-algorithms/consolidating-data/getting-started
#!pip install nasdaq-data-link
import nasdaqdatalink
def tradingapp3():
    # Get data
    #dataset = ql.get("CHRIS/CME_ES1", start_date="", end_date="", api_key='hiqkVfCKR65jR9y7yvoM')
    #dataset = ql.get("WIKI/AAPL", start_date="", end_date="", api_key='hiqkVfCKR65jR9y7yvoM')
    nasdaqdatalink.ApiConfig.api_key = 'hiqkVfCKR65jR9y7yvoM'
    #dataset = nasdaqdatalink.get_table('ZACKS/FC', ticker='AAPL')
    dataset = nasdaqdatalink.get_table('WIKI/PRICES', qopts = { 'columns': ['ticker', 'date', 'open', 'high', 'low', 'close'] }, ticker = ['AAPL'], date = { 'gte': '2016-01-01', 'lte': '2016-12-31' })
    #dataset = nasdaqdatalink.get('NSE/OIL')
    
    #dataset = nasdaqdatalink.get("CHRIS/CME_ES1", 
       #                      start_date="2023-01-01",  # Proporciona un rango de fechas válido
               #              end_date="2023-12-31")  # Usa tu API Key válida
    dataset = dataset.dropna()
   # dataset = dataset[['open', 'high', 'low', 'close']]
    dataset = dataset.set_index('date')
    dataset = dataset[['open', 'high', 'low', 'close']]
    dataset['H-L'] = dataset['close'] - dataset['open']
    dataset['O-C'] = dataset['high'] - dataset['low']
    dataset['10day MA'] = dataset['close'].shift(1).rolling(window=10).mean()
    dataset['20day MA'] = dataset['close'].shift(1).rolling(window=20).mean()
    dataset['55day MA'] = dataset['close'].shift(1).rolling(window=55).mean()
    dataset['stdev'] = dataset['close'].rolling(window=55).std()
    dataset['Price_Rise'] = np.where(dataset['close'].shift(-1) > dataset['close'], 1, 0)
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

    trade_dataset['Return_Tomorrow'] = np.log(trade_dataset['close'].shift(-1) / trade_dataset['close'])
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

import nasdaqdatalink
def tradingapp2():
    # Get data
    #dataset = ql.get("CHRIS/CME_ES1", start_date="", end_date="", api_key='hiqkVfCKR65jR9y7yvoM')
    #dataset = ql.get("WIKI/AAPL", start_date="", end_date="", api_key='hiqkVfCKR65jR9y7yvoM')
    nasdaqdatalink.ApiConfig.api_key = 'hiqkVfCKR65jR9y7yvoM'
    plot_file_relative = None  # Initialize plot file variable
    selected_ticker = None  # Initialize ticker variable

    if request.method == 'POST':
        # Get the selected stock ticker from the form
        selected_ticker = request.form.get('ticker')

        # Get data for the selected ticker
        dataset = nasdaqdatalink.get_table(
            'WIKI/PRICES',
            qopts={'columns': ['ticker', 'date', 'open', 'high', 'low', 'close']},
            ticker=[selected_ticker],
            date={'gte': '2016-01-01', 'lte': '2016-12-31'}
        )

        dataset = dataset.dropna()
        dataset = dataset.set_index('date')
        dataset = dataset[['open', 'high', 'low', 'close']]
        dataset['H-L'] = dataset['close'] - dataset['open']
        dataset['O-C'] = dataset['high'] - dataset['low']
        dataset['10day MA'] = dataset['close'].shift(1).rolling(window=10).mean()
        dataset['20day MA'] = dataset['close'].shift(1).rolling(window=20).mean()
        dataset['55day MA'] = dataset['close'].shift(1).rolling(window=55).mean()
        dataset['stdev'] = dataset['close'].rolling(window=55).std()
        dataset['Price_Rise'] = np.where(dataset['close'].shift(-1) > dataset['close'], 1, 0)
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

        trade_dataset['Return_Tomorrow'] = np.log(trade_dataset['close'].shift(-1) / trade_dataset['close'])
        trade_dataset['Strategy'] = np.where(trade_dataset['y_pred'] == 1, -trade_dataset['Return_Tomorrow'], trade_dataset['Return_Tomorrow'])
        trade_dataset['Market_Accumulated'] = trade_dataset['Return_Tomorrow'].cumsum()
        trade_dataset['Strategy_Accumulated'] = trade_dataset['Strategy'].cumsum()

        # Plot results
        plt.figure(figsize=(10, 5))
        plt.plot(trade_dataset['Market_Accumulated'], label='Market Accumulated')
        plt.plot(trade_dataset['Strategy_Accumulated'], label='Strategy Accumulated')
        plt.legend()

        # Save the plot to a file
        plot_file = 'static/plot.png'
        plt.savefig(plot_file)
        plt.close()

        # Generate the URL relative for the image of the plot
        plot_file_relative = 'static/plot.png'

    return render_template('trading.html', plot_file=plot_file_relative, ticker=selected_ticker)


##############################

def rifapp():
    if request.method == 'POST':
        title = request.form.get('title')
        repo_url = request.form.get('repo_url')

        # Standard deviations for normalization
        sigma_contributors_count = 2.556929
        sigma_owner_followers = 4109.995792
        sigma_forks_count = 83.046059 
        sigma_stargazers_count = 475.880867 
        sigma_citation_count = 227.967523

        # Get paper and author details from Semantic Scholar
        paper_id, paper_citation_count, authors = get_semanticscholar_info(title)
        
        # Get GitHub repository details
        repo_name, repo_html_url, stargazers_count, forks_count, language, owner_followers, contributors_count = get_github_info(repo_url)

        # Calculate the custom Research Impact Factor (RIF)
        rif = 0.5 * (int(paper_citation_count) if paper_citation_count != 'No citation info' else 0) / sigma_citation_count \
            + 0.45 * stargazers_count / sigma_stargazers_count \
            + 0.05 * forks_count / sigma_forks_count

        # Generate radar chart
        categories = ['Stars', 'Forks', 'Citations', 'Owner Followers', 'Contributors']
        values = [stargazers_count, forks_count, int(paper_citation_count) if paper_citation_count != 'No citation info' else 0, owner_followers, contributors_count]
        
        # close any existing plot
        plt.close()

        # Create radar plot
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        angles = [n / float(len(categories)) * 2 * 3.14159 for n in range(len(categories))]
        values += values[:1]
        angles += angles[:1]
        
        ax.fill(angles, values, color='blue', alpha=0.25)
        ax.plot(angles, values, color='blue', linewidth=2)
        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        
        # Convert plot to PNG in base64 format for embedding in HTML
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')

        # Render results with the radar chart and table of authors
        #return render_template('result_rif.htm', title=title, authors=authors, repo_name=repo_name, forks=forks_count, stars=stargazers_count, contributors=contributors_count, citations=paper_citation_count, followers=owner_followers, factor=rif, plot_url=plot_url)

    #return render_template('rif.htm')
        return render_template('rift.html', title=title, authors=authors, repo_name=repo_name, forks=forks_count, stars=stargazers_count, contributors=contributors_count, citations=paper_citation_count, followers=owner_followers, factor=rif, plot_url=plot_url)

    return render_template('rift.html')



# Function to get detailed author data by their IDs
def get_author_data_by_ids(author_ids):
    url = 'https://api.semanticscholar.org/graph/v1/author/batch'
    query_params = {
        'fields': 'name,paperCount,citationCount,hIndex'
    }
    response = requests.post(url, params=query_params, json={"ids": author_ids})

    if response.status_code == 200:
        return response.json()  # Return response in JSON format
    else:
        print(f"Error: request failed with status code {response.status_code}")
        return None

# Function to get paper details and author information from Semantic Scholar by title
def get_semanticscholar_info(title):
    try:
        # Search for the paper using the title
        url = f"https://api.semanticscholar.org/graph/v1/paper/search/match?query={title}&fields=paperId,citationCount,authors"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json().get('data', [])
            if data:
                # Extract paper information (citation count for the paper)
                paper_id = data[0].get('paperId', 'No paperId')
                paper_citation_count = data[0].get('citationCount', 'No citation info')

                # Extract authors and fetch their individual citation, paper counts, and h-index
                authors_info = []
                author_ids = [author['authorId'] for author in data[0].get('authors', [])]
                if author_ids:
                    authors_data = get_author_data_by_ids(author_ids)
                    if authors_data:
                        for author in authors_data:
                            authors_info.append({
                                'name': author.get('name', 'N/A'),
                                'papers': author.get('paperCount', 'N/A'),
                                'citations': author.get('citationCount', 'N/A'),
                                'h_index': author.get('hIndex', 'N/A')
                            })
                # Return both paper citation count and detailed author data
                return paper_id, paper_citation_count, authors_info

        return 'No paperId', 'No citation info', []

    except Exception as e:
        print(f"Error fetching Semantic Scholar data for {title}: {e}")
        return 'Error', 'Error', []

# Function to get GitHub repository details
def get_github_info(repo_url):
    
    try:
        if not repo_url.startswith('https://github.com/'):
            print(f"Invalid repository URL: {repo_url}")
            return 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'

        repo_path = repo_url.replace('https://github.com/', '').strip()
        api_url = f"https://api.github.com/repos/{repo_path}"
        headers = {'Accept': 'application/vnd.github+json'}
        response = requests.get(api_url, headers=headers)

        if response.status_code == 200:
            repo = response.json()
            owner = repo['owner']['login']
            owner_url = f"https://api.github.com/users/{owner}"
            owner_response = requests.get(owner_url, headers=headers)
            owner_followers = owner_response.json().get("followers", 0) if owner_response.status_code == 200 else "N/A"

            contributors_url = f"https://api.github.com/repos/{repo_path}/contributors"
            contributors_response = requests.get(contributors_url, headers=headers)
            contributors_count = len(contributors_response.json()) if contributors_response.status_code == 200 else "N/A"

            return repo.get('full_name', ''), repo.get('html_url', ''), repo.get('stargazers_count', 0), repo.get('forks_count', 0), repo.get('language', ''), owner_followers, contributors_count
        r=response.status_code
        print(f"Error 200 retrieving GitHub data for {repo_url}: {r}")
        return 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A' , 'N/A'

    except Exception as e:
        print(f"Error retrieving GitHub data for {repo_url}: {e}")
        return 'Error', 'Error', 'Error', 'Error', 'Error', 'Error'
    

#########blockchain certificates######

import datetime
import hashlib
import json

# Blockchain class
class Blockchain:
    
    def __init__(self):
        self.chain = []
        self.certificates = []  # Store certificate hashes
        self.create_block(proof=1, previous_hash='0')

    def create_block(self, proof, previous_hash, certificate_hash=None):
        block = {
            'index': len(self.chain) + 1,
            'timestamp': str(datetime.datetime.now()),
            'proof': proof,
            'previous_hash': previous_hash,
            'certificate_hash': certificate_hash  # Store the certificate hash
        }
        self.chain.append(block)
        return block

    def get_previous_block(self):
        return self.chain[-1]

    def proof_of_work(self, previous_proof):
        new_proof = 1
        check_proof = False
        while not check_proof:
            hash_operation = hashlib.sha256(str(new_proof**2 - previous_proof**2).encode()).hexdigest()
            if hash_operation[:4] == '0000':
                check_proof = True
            else:
                new_proof += 1
        return new_proof

    def hash(self, block):
        encoded_block = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(encoded_block).hexdigest()

    def add_certificate(self, certificate_hash):
        if certificate_hash in self.certificates:  # Check if certificate already exists
            return None  # Indicate that the certificate already exists

        # If it doesn't exist, add the certificate to the list
        self.certificates.append(certificate_hash)
        previous_block = self.get_previous_block()
        previous_proof = previous_block['proof']
        proof = self.proof_of_work(previous_proof)
        previous_hash = self.hash(previous_block)
        block = self.create_block(proof, previous_hash, certificate_hash)
        return block

    def validate_certificate(self, certificate_hash):
        for block in self.chain:
            if block.get('certificate_hash') == certificate_hash:
                return block
        return None


# Flask app initialization
#app = Flask(__name__)

# Create blockchain instance
blockchain = Blockchain()

# Route for the main page with forms
#@app.route('/', methods=['GET', 'POST'])
def app_certificates():
# Create blockchain instance
   
    if request.method == 'POST':
        if 'certificate_hash' in request.form:  # Add Certificate Form Submission
            certificate_hash = request.form.get('certificate_hash')
            if not certificate_hash:
                return render_template('form2.html', add_message='Certificate hash is required')
            
            # Add the certificate to the blockchain
            block = blockchain.add_certificate(certificate_hash)
            if block:
                add_message = f'Success! Certificate added at block {block["index"]}, on {block["timestamp"]}.'
            else:
                add_message = 'This certificate already exists in the blockchain.'
            return render_template('form2.html', add_message=add_message)

        elif 'certificate_hash_validate' in request.form:  # Validate Certificate Form Submission
            certificate_hash = request.form.get('certificate_hash_validate')
            if not certificate_hash:
                return render_template('form2.html', validate_message='Certificate hash is required')

            # Validate the certificate
            block = blockchain.validate_certificate(certificate_hash)
            if block:
                validate_message = f'Certificate found in block {block["index"]}, timestamp: {block["timestamp"]}.'
            else:
                validate_message = 'This certificate does not exist in the blockchain.'
            return render_template('form2.html', validate_message=validate_message)
    
    # On GET request, show the form with no messages
    return render_template('form2.html')