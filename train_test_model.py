import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the data
df = pd.read_csv("filtered_data.csv")

# Initialize the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Define the rolling window for training and testing
window_size = 8  # e.g., train on 8 years, test on the next year
prediction_year = 2015

# Initialize lists to store predictions and actual values
all_predictions = []
all_actuals = []

# Group the dataset by country
grouped_df = df.groupby('Country')

# Loop through each country group
for country, data in grouped_df:
    print(f"Processing data for {country}...")
    # Extract features and target variable
    X = data.drop(columns=['Country', 'Year', 'Health_Expenditure_as_GDP_Percentage'])
    y = data['Health_Expenditure_as_GDP_Percentage']
    # Loop through each year for training within the country group
    for i in range(len(data) - window_size):
        # Extract the training data for the current window
        X_train = X.iloc[i:i + window_size]
        y_train = y.iloc[i + window_size]  # Convert to scalar value

        # Extract the testing data for the next year
        X_test = X.iloc[i + window_size:i + window_size + 1]
        y_test = y.iloc[i + window_size:i + window_size + 1]

        # Train the model
        model.fit(X_train, [y_train])  # Ensure y_train is a list

        # Make predictions for the next year
        y_pred = model.predict(X_test)

        # Store predictions and actual values
        all_predictions.append(y_pred[0])  # Access the single prediction
        all_actuals.append(y_test)

# Evaluate the model (e.g., using mean squared error)
mse = mean_squared_error(all_actuals, all_predictions)
print(f"Mean Squared Error: {mse}")

# Optionally, you can visualize the predictions and actual values to assess the model's performance
