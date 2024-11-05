# simulate_2022.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

def load_data():
    df = pd.read_csv('covid_data.csv', parse_dates=['date'])
    df = df.sort_values('date')
    return df

def preprocess_data(df):
    scaler = MinMaxScaler()
    data = df['positive'].values.reshape(-1, 1)
    scaled_data = scaler.fit_transform(data)
    
    lookback = 30
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, epochs=50, batch_size=32):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model

def predict_future_cases(model, scaler, last_sequence, days=365):
    predictions = []
    current_sequence = last_sequence
    
    for _ in range(days):
        pred = model.predict(current_sequence.reshape(1, -1, 1), verbose=0)
        predictions.append(pred[0, 0])
        current_sequence = np.append(current_sequence[1:], pred)
    
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions.flatten()

def plot_predictions(df, predictions, start_date):
    plt.figure(figsize=(15, 7))
    plt.plot(df['date'], df['positive'], label='Historical Data')
    
    future_dates = pd.date_range(start=start_date, periods=len(predictions))
    plt.plot(future_dates, predictions, label='Predicted Cases for 2022', color='red', linestyle='--')
    
    plt.title("COVID-19 Case Predictions for 2022")
    plt.xlabel("Date")
    plt.ylabel("Total Cases")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def run_simulation_2022():
    df = load_data()
    X, y, scaler = preprocess_data(df)
    model = build_lstm_model((X.shape[1], 1))
    train_model(model, X, y, epochs=50, batch_size=32)
    
    last_sequence = X[-1]
    future_predictions = predict_future_cases(model, scaler, last_sequence, days=365)
    
    plot_predictions(df, future_predictions, start_date=df['date'].max() + pd.Timedelta(days=1))
    return future_predictions