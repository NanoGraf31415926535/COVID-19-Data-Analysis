# predict.py
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

DATA_FILE = "covid_data.csv"

def load_data():
    """Load and prepare data for prediction"""
    df = pd.read_csv(DATA_FILE, parse_dates=['date'])
    return df

def prepare_state_data(df, state='CA'):
    """Prepare data for a specific state"""
    state_df = df[df['state'] == state].copy()
    state_df = state_df.sort_values('date')
    state_df['day_num'] = (state_df['date'] - state_df['date'].min()).dt.days
    return state_df[['day_num', 'positive']].dropna()

def build_model(df, state='CA'):
    """Build and train the linear regression model"""
    state_df = prepare_state_data(df, state)
    X = state_df[['day_num']]
    y = state_df['positive']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Model Mean Squared Error: {mse}")
    
    return model

def predict_future(model, df, state='CA', days_ahead=30):
    """Predict future cases"""
    state_df = prepare_state_data(df, state)
    last_day = state_df['day_num'].max()
    future_days = np.array([[last_day + i] for i in range(1, days_ahead+1)])
    predictions = model.predict(future_days)
    print("Future predictions (30 days ahead):", predictions)
    return predictions

def main():
    df = load_data()
    model = build_model(df)
    predictions = predict_future(model, df)
    return predictions

if __name__ == "__main__":
    main()