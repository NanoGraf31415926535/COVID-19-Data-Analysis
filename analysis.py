# analysis.py
import pandas as pd

DATA_FILE = "covid_data.csv"

def load_data():
    # Load the data from CSV file
    df = pd.read_csv(DATA_FILE, parse_dates=['date'])
    return df

def calculate_daily_trends(df):
    # Calculate daily new cases, deaths, and other key metrics
    df = df.sort_values(by=['state', 'date'])
    df['new_cases'] = df.groupby('state')['positive'].diff().fillna(0)
    df['new_deaths'] = df.groupby('state')['death'].diff().fillna(0)
    print("Daily trends calculated.")
    return df

def summarize_state_data(df):
    # Example analysis for state-wise summaries
    latest_date = df['date'].max()
    state_summary = df[df['date'] == latest_date].groupby('state').agg({
        'positive': 'sum',
        'death': 'sum',
        'hospitalizedCurrently': 'sum',
        'inIcuCurrently': 'sum',
        'onVentilatorCurrently': 'sum'
    })
    print("State-wise summary calculated.")
    return state_summary

def main():
    df = load_data()
    df = calculate_daily_trends(df)
    state_summary = summarize_state_data(df)
    print(state_summary.head())  # Display a preview of the state summary

if __name__ == "__main__":
    main()