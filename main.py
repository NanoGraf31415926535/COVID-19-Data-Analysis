# main.py

import data_fetch
import analysis
import visualize
import predict
import deep_learning
import simulate_2022  # Import the new simulation module

def main():
    print("Starting COVID-19 Data Analysis Project...")
    
    # Fetch and preprocess data
    data_fetch.main()
    
    # Perform analysis
    analysis.main()
    
    # Traditional predictions
    df = predict.load_data()
    model = predict.build_model(df)
    linear_predictions = predict.predict_future(model, df)
    
    # Deep Learning predictions
    print("\nInitializing Deep Learning Model...")
    predictor = deep_learning.CovidPredictor()
    predictor.train(df, epochs=50)
    future_predictions_dl = predictor.predict_future(df)
    
    # Visualization
    visualize.main(linear_predictions)
    predictor.plot_predictions(df, predictions=future_predictions_dl)
    
    # Run the 2022 simulation
    print("\nRunning 2022 COVID-19 Simulation with LSTM...")
    future_predictions_2022 = simulate_2022.run_simulation_2022()
    
    # Generate comprehensive report
    generate_combined_report(linear_predictions, future_predictions_dl, future_predictions_2022)
    
    print("Project completed successfully.")

def generate_combined_report(linear_predictions, dl_predictions, simulation_predictions):
    """Generate a report comparing prediction methods, including 2022 simulation"""
    with open("covid_analysis_report.txt", "w") as f:
        f.write("COVID-19 Comprehensive Analysis Report\n")
        f.write("="*40 + "\n\n")
        
        f.write("1. Linear Regression Predictions:\n")
        f.write("-"*30 + "\n")
        f.write(f"30-Day Range: {linear_predictions.min():,.0f} - {linear_predictions.max():,.0f}\n")
        f.write(f"Final Prediction: {linear_predictions[-1]:,.0f}\n\n")
        
        f.write("2. Deep Learning (LSTM) Predictions:\n")
        f.write("-"*30 + "\n")
        f.write(f"30-Day Range: {dl_predictions.min():,.0f} - {dl_predictions.max():,.0f}\n")
        f.write(f"Final Prediction: {dl_predictions[-1]:,.0f}\n\n")
        
        f.write("3. 2022 Simulation Predictions:\n")
        f.write("-"*30 + "\n")
        f.write(f"365-Day Range: {simulation_predictions.min():,.0f} - {simulation_predictions.max():,.0f}\n")
        f.write(f"Final Prediction: {simulation_predictions[-1]:,.0f}\n\n")
        
        f.write("Note: The 2022 simulation provides an extended forecast for future cases.\n")

if __name__ == "__main__":
    main()
