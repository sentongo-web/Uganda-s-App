import pandas as pd
import joblib
import streamlit as st
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ---- DATA LOADING ----
@st.cache_data
def load_data():
    df = pd.read_csv('data/Uganda_imports_train.csv')
    df['Value_Density'] = df['CIF_Value_USD'] / (df['Gross_Mass_kg'] + 1e-6)
    df['Tax_Load'] = df['Tax_Rate'] * df['CIF_Value_USD']
    df['Import_Duration'] = df['Year'] + df['Month'] / 12
    df['FOB_per_kg'] = df['FOB_Value_USD'] / (df['Gross_Mass_kg'] + 1e-6)
    df['Freight_per_kg'] = df['Freight_USD'] / (df['Gross_Mass_kg'] + 1e-6)
    df['Insurance_per_kg'] = df['Insurance_USD'] / (df['Gross_Mass_kg'] + 1e-6)
    return df

@st.cache_resource
def load_model():
    model_path = 'models/best_price_predictor.pkl'
    if not os.path.exists(model_path):
        st.error("Model file not found. Please ensure it exists under /models.")
        return None, None
    model = joblib.load(model_path)
    return model['model'], model['preprocessor']

# Columns for preprocessing
TARGET = 'Unit_Price_UGX'

def preprocess_data(data):
    processed = load_data()
    to_drop = ['CIF_Value_UGX', 'Invoice_Amount', 'Value_per_kg', 'Value_per_unit', 'Date']
    return processed.drop(columns=to_drop, errors='ignore')

numeric_features = [
    'Quantity', 'Net_Mass_kg', 'Gross_Mass_kg',
    'FOB_Value_USD', 'Freight_USD', 'Insurance_USD',
    'CIF_Value_USD', 'Tax_Rate', 'Tax_Load',
    'Import_Duration', 'Value_Density',
    'FOB_per_kg', 'Freight_per_kg', 'Insurance_per_kg'
]

categorical_features = [
    'HS_Code', 'Country_of_Origin', 'Port_of_Shipment',
    'Quantity_Unit', 'Currency_Code', 'Mode_of_Transport',
    'Valuation_Method', 'Item_Description'
]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ]
)
