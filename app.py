import random
from flask import Flask, render_template, redirect, url_for, session, flash,request
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, ValidationError
import bcrypt
from flask_mysqldb import MySQL
from flask import request, jsonify
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
import joblib
import nltk
import numpy as np
import pickle
# Ensure required NLTK data is downloaded
nltk.download('stopwords')
nltk.download('punkt')


#adhd test
# Load the model and label encoder from the same pickle file
with open('adhd.pkl', 'rb') as f:
    model, le = pickle.load(f)

# Load feature names from your training data (use the same CSV used during training)
df = pd.read_csv('adhd.csv') # Replace with your actual CSV file
features = list(df.drop('Disorder', axis=1).columns)

# Disorders that are considered NOT ADHD
non_adhd_disorders = ['ASD', 'Anxiety', 'Bipolar Disorder', 'MDD', 'PDD', 'PTSD']

@app.route('/adhd')
def index():
    return render_template('adhdQuize.html', features=features)

@app.route('/predictAdhd', methods=['POST'])
def predict_adhd():
    data = request.json  # Expecting a dict of {feature_name: 0 or 1}
    
    input_df = pd.DataFrame([data])
    prediction = model.predict(input_df)
    predicted_disorder = le.inverse_transform(prediction)[0]

    if predicted_disorder in non_adhd_disorders:
        result = "You are NOT suffering from ADHD."
    else:
        result = "You ARE suffering from ADHD."

    return jsonify({'result': result})

