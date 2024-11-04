# visualize.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_FILE = "covid_data.csv"

def load_data():
    # Load the data
    df = pd.read_csv(DATA_FILE, parse_dates=['date'])
    return df

def plot_daily_cases(df, state='CA'):
    # Filter data for the selected state
    state_df = df[df['state'] == state].copy()
    state_df['new_cases'] = state_df['positive'].diff().fillna(0)
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=state_df, x='date', y='new_cases')
    plt.title(f"Daily New Cases in {state}")
    plt.xlabel("Date")
    plt.ylabel("New Cases")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_statewise_summary(df):
    latest_date = df['date'].max()
    summary_df = df[df['date'] == latest_date]
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=summary_df, x='state', y='positive')
    plt.xticks(rotation=90)
    plt.title(f"State-wise Positive Cases as of {latest_date.strftime('%Y-%m-%d')}")
    plt.xlabel("State")
    plt.ylabel("Positive Cases")
    plt.tight_layout()
    plt.show()

def plot_predictions(df, predictions=None):
    if predictions is not None:
        # Plot historical data and predictions for CA
        ca_data = df[df['state'] == 'CA'].copy()
        ca_data = ca_data.sort_values('date')
        
        plt.figure(figsize=(12, 6))
        plt.plot(ca_data['date'], ca_data['positive'], label='Historical Data')
        
        # Create future dates for predictions
        last_date = ca_data['date'].max()
        future_dates = pd.date_range(start=last_date, periods=31)[1:]
        plt.plot(future_dates, predictions, 'r--', label='Predictions')
        
        plt.title('COVID-19 Cases in California - Historical and Predicted')
        plt.xlabel('Date')
        plt.ylabel('Total Cases')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def main(predictions=None):
    df = load_data()
    plot_daily_cases(df)
    plot_statewise_summary(df)
    if predictions is not None:
        plot_predictions(df, predictions)

if __name__ == "__main__":
    main()