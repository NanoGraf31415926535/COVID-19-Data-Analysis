# main.py
import data_fetch
import analysis
import visualize
import predict
import deep_learning

def main():
    print("Starting COVID-19 Data Analysis Project...")
    
    # Fetch and preprocess data
    data_fetch.main()
    
    # Perform analysis
    analysis.main()
    
    # Get traditional predictions
    df = predict.load_data()
    model = predict.build_model(df)
    linear_predictions = predict.predict_future(model, df)
    
    # Get deep learning predictions
    print("\nInitializing Deep Learning Model...")
    predictor = deep_learning.CovidPredictor()
    
    try:
        # Train the model and get predictions
        predictor.train(df, epochs=50)
        future_predictions = predictor.predict_future(df)
        
        # Visualize both predictions
        visualize.main(linear_predictions)
        predictor.plot_predictions(df, predictions=future_predictions)
        
        # Generate comprehensive report
        generate_combined_report(linear_predictions, future_predictions)
        
    except Exception as e:
        print(f"Error in deep learning analysis: {str(e)}")
        print("Continuing with linear predictions only...")
        
    print("Project completed successfully.")

def generate_combined_report(linear_predictions, dl_predictions):
    """Generate a report comparing both prediction methods"""
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
        
        f.write("3. Comparison Analysis:\n")
        f.write("-"*30 + "\n")
        difference = abs(linear_predictions[-1] - dl_predictions[-1])
        f.write(f"Prediction Difference: {difference:,.0f} cases\n")
        f.write(f"Percentage Difference: {(difference/linear_predictions[-1]*100):.2f}%\n\n")
        
        f.write("Note: Deep learning predictions may be more accurate for complex patterns\n")
        f.write("but require more computational resources and training data.\n")

if __name__ == "__main__":
    main()