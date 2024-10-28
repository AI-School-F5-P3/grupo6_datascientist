import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


# Define the model (same as in your original code)
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=100, kernel_size=3, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=100, out_channels=80, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=80, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 26 * 26, 500)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.pool(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.pool(x)
        x = nn.functional.relu(self.conv3(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = nn.functional.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        return x

def load_model_image(model_path):
    model = CNNModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    image_array = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()
    return image_array.unsqueeze(0)

def predict_image(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
    probability = output.item()
    predicted_class = "Stroke" if probability > 0.5 else "Normal"
    return predicted_class, probability


def print_outcome(predicted_class, probability):
    STROKE_STYLE = "padding: 20px; background-color: #f44336; color: white; margin-bottom: 15px; text-align: center; font-size: 24px; width: 80%"
    HEALTHY_STYLE = "padding: 20px; background-color: #4cbb17; color: white; margin-bottom: 15px; text-align: center; font-size: 24px; width: 80%"
    if predicted_class == "Stroke":
        st.write(f'<div style="{STROKE_STYLE}">Ictus</div>', unsafe_allow_html=True)
    else:
        st.write(f'<div style="{HEALTHY_STYLE}">Normal</div>', unsafe_allow_html=True)
    st.write(f"Probability: {probability:.2f}")

def print_error(error):
    ERROR_STYLE = "padding: 20px; background-color: #ffc300; color: white; margin-bottom: 15px; text-align: center; font-size: 24px;"
    st.write(f'<div style="{ERROR_STYLE}">Error: {error}</div>', unsafe_allow_html=True)



# Custom transformer for feature engineering
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        X_['age_group'] = pd.cut(X_['age'], bins=[0, 18, 30, 45, 60, 75, 100], labels=['0-18', '19-30', '31-45', '46-60', '61-75', '75+'])
        X_['bmi_category'] = pd.cut(X_['bmi'], bins=[0, 18.5, 25, 30, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        X_['glucose_category'] = pd.cut(X_['avg_glucose_level'], bins=[0, 70, 100, 125, 200, 300], labels=['Low', 'Normal', 'Prediabetes', 'Diabetes', 'High'])
        X_['age_glucose_interaction'] = X_['age'] * X_['avg_glucose_level']
        X_['bmi_age_interaction'] = X_['bmi'] * X_['age']
        X_['health_score'] = X_['hypertension'] + X_['heart_disease'] + (X_['bmi'] > 30).astype(int) + (X_['age'] > 60).astype(int)
        return X_


