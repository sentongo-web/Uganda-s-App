import pandas as pd
import joblib
import streamlit as st
import os
import requests
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ---- DATA LOADING ----
@st.cache_data
def load_data():
    """Load and preprocess data for dashboard visualizations"""
    df = pd.read_csv('data/Uganda_imports_train.csv')
    
    # Create essential features for reports
    df['Value_Density'] = df['CIF_Value_USD'] / (df['Gross_Mass_kg'] + 1e-6)
    df['Tax_Load'] = df['Tax_Rate'] * df['CIF_Value_USD']
    df['Import_Duration'] = df['Year'] + df['Month'] / 12
    df['FOB_per_kg'] = df['FOB_Value_USD'] / (df['Gross_Mass_kg'] + 1e-6)
    df['Freight_per_kg'] = df['Freight_USD'] / (df['Gross_Mass_kg'] + 1e-6)
    df['Insurance_per_kg'] = df['Insurance_USD'] / (df['Gross_Mass_kg'] + 1e-6)
    
    return df

# ---- MODEL LOADING ----
@st.cache_resource
def load_model():
    """Load trained model and preprocessor"""
    model_path = 'best_price_predictor.pkl'

    if not os.path.exists(model_path):
        st.info("Downloading model...")

        # --- Google Drive large file download with confirmation token ---
        def download_file_from_google_drive(file_id, destination):
            URL = "https://docs.google.com/uc?export=download"

            session = requests.Session()
            response = session.get(URL, params={'id': file_id}, stream=True)
            token = get_confirm_token(response)

            if token:
                params = {'id': file_id, 'confirm': token}
                response = session.get(URL, params=params, stream=True)

            save_response_content(response, destination)

        def get_confirm_token(response):
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    return value
            return None

        def save_response_content(response, destination):
            CHUNK_SIZE = 32768

            with open(destination, "wb") as f:
                for chunk in response.iter_content(CHUNK_SIZE):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)

        # Your Google Drive file ID
        file_id = '1tkHVw8_7J3UcI652ef_tboY7gUucEwVs'
        download_file_from_google_drive(file_id, model_path)

    # Load model
    model = joblib.load(model_path)
    return model['model'], model['preprocessor']

# ---- MODELING PREPROCESSING ----
TARGET = 'Unit_Price_UGX'

def preprocess_data(data):
    """Feature engineering for model training/prediction"""
    processed = load_data()

    to_drop = ['CIF_Value_UGX', 'Invoice_Amount', 'Value_per_kg', 
               'Value_per_unit', 'Date']
    
    return processed.drop(columns=to_drop, errors='ignore')

# ColumnTransformer setup
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
