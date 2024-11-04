# data_fetch.py
import requests
import pandas as pd
import os
from datetime import datetime

API_URL = "https://api.covidtracking.com/v1/states/daily.json"  # Endpoint for historical data
DATA_FILE = "covid_data.csv"

def fetch_data():
    print("Fetching data from API...")
    response = requests.get(API_URL)
    if response.status_code == 200:
        data = response.json()
        print("Data fetched successfully.")
        return data
    else:
        print("Failed to fetch data.")
        return None

def preprocess_data(data):
    # Convert data to DataFrame and select relevant columns
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df = df[['date', 'state', 'positive', 'negative', 'hospitalizedCurrently', 
             'inIcuCurrently', 'onVentilatorCurrently', 'death']]
    df = df.fillna(0)  # Fill missing values with zero
    print("Data preprocessing completed.")
    return df

def save_data(df):
    # Save data to a CSV file
    if not os.path.exists(DATA_FILE):
        df.to_csv(DATA_FILE, index=False)
        print(f"Data saved to {DATA_FILE}.")
    else:
        print(f"{DATA_FILE} already exists. Overwriting with updated data.")
        df.to_csv(DATA_FILE, index=False)

def main():
    data = fetch_data()
    if data:
        df = preprocess_data(data)
        save_data(df)

if __name__ == "__main__":
    main()