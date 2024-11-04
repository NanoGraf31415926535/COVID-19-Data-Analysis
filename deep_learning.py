# deep_learning.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

class CovidPredictor:
    def __init__(self, lookback=30):
        self.lookback = lookback
        self.model = None
        self.scaler = MinMaxScaler()
        
    def create_sequences(self, data):
        """Create sequences for LSTM model"""
        X, y = [], []
        for i in range(len(data) - self.lookback):
            X.append(data[i:(i + self.lookback)])
            y.append(data[i + self.lookback])
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """Build LSTM model"""
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    def prepare_data(self, df, state='CA'):
        """Prepare data for the model"""
        try:
            # Filter data for specific state and sort by date
            state_data = df[df['state'] == state].sort_values('date')
            
            if state_data.empty:
                raise ValueError(f"No data found for state {state}")
            
            # Get daily cases
            daily_cases = state_data['positive'].values.reshape(-1, 1)
            
            # Scale the data
            scaled_data = self.scaler.fit_transform(daily_cases)
            
            # Create sequences
            X, y = self.create_sequences(scaled_data)
            
            # Split into train and test
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            return X_train, X_test, y_train, y_test
            
        except KeyError as e:
            print(f"Error: Missing column in dataset. Available columns: {df.columns.tolist()}")
            raise e
        except Exception as e:
            print(f"Error preparing data: {str(e)}")
            raise e

    def train(self, df, state='CA', epochs=50, batch_size=32):
        """Train the model"""
        try:
            # Prepare data
            X_train, X_test, y_train, y_test = self.prepare_data(df, state)
            
            # Build model if not already built
            if self.model is None:
                self.model = self.build_model((self.lookback, 1))
            
            # Train model
            print("\nTraining LSTM model...")
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                verbose=1
            )
            
            # Make predictions for validation
            test_predictions = self.model.predict(X_test)
            
            # Inverse transform predictions
            test_predictions = self.scaler.inverse_transform(test_predictions)
            y_test_inv = self.scaler.inverse_transform(y_test.reshape(-1, 1))
            
            # Calculate and print MSE
            mse = np.mean((test_predictions - y_test_inv) ** 2)
            print(f"\nLSTM Model MSE: {mse:,.2f}")
            
            return test_predictions.flatten()
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            raise e

    def predict_future(self, df, state='CA', days_ahead=30):
        """Predict future cases"""
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
            
        try:
            # Get the last sequence of data
            state_data = df[df['state'] == state].sort_values('date')
            last_sequence = state_data['positive'].values[-self.lookback:].reshape(-1, 1)
            last_sequence = self.scaler.transform(last_sequence)
            
            # Make predictions
            predictions = []
            current_sequence = last_sequence.copy()
            
            for _ in range(days_ahead):
                # Predict next value
                next_pred = self.model.predict(current_sequence.reshape(1, self.lookback, 1), verbose=0)
                predictions.append(next_pred[0, 0])
                
                # Update sequence
                current_sequence = np.roll(current_sequence, -1)
                current_sequence[-1] = next_pred
            
            # Inverse transform predictions
            predictions = np.array(predictions).reshape(-1, 1)
            predictions = self.scaler.inverse_transform(predictions)
            
            return predictions.flatten()
            
        except Exception as e:
            print(f"Error making predictions: {str(e)}")
            raise e

    def plot_predictions(self, df, state='CA', predictions=None):
        """Plot historical data and predictions"""
        try:
            state_data = df[df['state'] == state].sort_values('date')
            
            plt.figure(figsize=(15, 7))
            
            # Plot historical data
            plt.plot(state_data['date'], state_data['positive'], 
                    label='Historical Data', color='blue')
            
            if predictions is not None:
                # Create future dates
                last_date = state_data['date'].max()
                future_dates = pd.date_range(start=last_date, periods=len(predictions)+1)[1:]
                
                # Plot predictions
                plt.plot(future_dates, predictions, 
                        label='LSTM Predictions', color='red', linestyle='--')
            
            plt.title(f'COVID-19 Cases in {state}: Historical Data and LSTM Predictions')
            plt.xlabel('Date')
            plt.ylabel('Total Cases')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error plotting predictions: {str(e)}")
            raise e

def main():
    # Load data
    df = pd.read_csv('covid_data.csv', parse_dates=['date'])
    
    # Initialize predictor
    predictor = CovidPredictor(lookback=30)
    
    # Train model
    print("Training LSTM model...")
    predictions = predictor.train(df, epochs=50)
    
    # Make future predictions
    future_predictions = predictor.predict_future(df)
    
    # Plot results
    predictor.plot_predictions(df, predictions=future_predictions)
    
    return future_predictions

if __name__ == "__main__":
    main()