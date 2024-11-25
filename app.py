# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 14:07:18 2024

@author: Lab
"""
import os
from flask import Flask, request, render_template
import numpy as np
import pickle
from itertools import product

# Initialize Flask app
app = Flask(__name__)

# Load the pickled model
filename = 'model.pkl'  # Path to the saved pickle model
try:
    with open(filename, 'rb') as file:
        loaded_model = pickle.load(file)
except Exception as e:
    raise ValueError(f"Failed to load the model from {filename}: {e}")

# Function to generate k-mers
def generate_kmers(sequence, k):
    return [sequence[i:i + k] for i in range(len(sequence) - k + 1)]

# Function to encode sequences using k-mer frequency
def kmer_encode(sequences, k):
    possible_kmers = [''.join(p) for p in product('ACGT', repeat=k)]
    kmer_vectors = []

    for sequence in sequences:
        kmers = generate_kmers(sequence, k)
        kmer_freq = {kmer: 0 for kmer in possible_kmers}
        for kmer in kmers:
            if kmer in kmer_freq:
                kmer_freq[kmer] += 1
        total_kmers = len(kmers) if kmers else 1  # Avoid division by zero
        kmer_vector = [count / total_kmers for count in kmer_freq.values()]
        kmer_vectors.append(kmer_vector)

    return np.array(kmer_vectors)

# Prediction function
def predict_promoter(sequence, model, k):
    encoded_sequence = kmer_encode([sequence], k)
    prediction = model.predict(encoded_sequence)  # Assuming a non-Keras model
    return 'Promoter' if prediction[0] == 1 else 'NON-Promoter'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sequences = []

    # Check if a sequence was entered directly
    if 'sequence' in request.form and request.form['sequence'].strip():
        sequences.append(request.form['sequence'].strip().upper())

    # Check if a file was uploaded
    if 'file' in request.files:
        file = request.files['file']
        if file and file.filename.endswith('.txt'):
            sequences.extend([line.decode('utf-8').strip().upper() for line in file if line.strip()])

    if not sequences:
        return render_template('index.html', error="Please enter a sequence or upload a valid file.")

    # Make predictions
    results = [(seq, predict_promoter(seq, loaded_model, k=6)) for seq in sequences]

    # Count promoters and non-promoters
    promoters_count = sum(1 for _, result in results if result == 'Promoter')
    non_promoters_count = len(results) - promoters_count

    return render_template('index.html', 
                           results=results, 
                           promoters_count=promoters_count, 
                           non_promoters_count=non_promoters_count)

if __name__ == '__main__':
    app.run(debug=True)
