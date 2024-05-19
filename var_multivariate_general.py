import pickle

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error

# Step 1: Load the data
df = pd.read_csv('data/time.csv')
df.dropna()

# Step 2: Prepare the data (if needed)

# Step 3: Filter data for the period 1995-2013
train_data = df[(df['Year'] >= 1995) & (df['Year'] <= 2013)]

# Step 4: Concatenate data for all countries' 25-year sequences
all_statistics = ['Air_Pollution_Deaths', 'Infectious_Disease_Deaths',
                  'Life_Expectancy',
                  'Smoking_Attributed_Deaths', 'Under_Fifteen_Mortality',
                  'Ozone_Pollution_Deaths']

# Initialize an empty DataFrame to store concatenated data
concatenated_data = pd.DataFrame(columns=all_statistics)

for country in df['Country'].unique():
    # Filter data for the current country
    country_data = train_data[train_data['Country'] == country]

    # Check if there are at least 19 years of data for the country (1995-2013)
    if len(country_data) >= 19:
        # Concatenate data for the years 1995-2013
        data_1995_2013 = country_data[all_statistics]
        concatenated_data = pd.concat([concatenated_data, data_1995_2013], axis=0)

# Step 5: Fit VAR model on the concatenated data
concatenated_data.dropna(inplace=True)
model = VAR(concatenated_data)
fitted_model = model.fit()

# Step 6: Store the fitted VAR model in a pickle file
with open('data/pkl/var_general.pkl', 'wb') as f:
    pickle.dump(fitted_model, f)

# Step 6: Test the model on 2014 data
test_data = df[df['Year'] == 2014]

# Initialize lists to store predictions and actual values
predictions = []
actuals = []

for country in df['Country'].unique():
    test_subset = test_data[test_data['Country'] == country][all_statistics]

    # Check if there are enough data points for testing
    if not test_subset.isna().any().any():
        # Predict 2014 values
        prediction = fitted_model.forecast(test_subset.values, steps=len(test_subset))
        predictions.append(prediction)
        actuals.append(test_subset.values)

    else:
        print(f"No data available for {country} in 2014")

# Scale the values
scaler = StandardScaler()
scaled_predictions = scaler.fit_transform(np.concatenate(predictions))
scaled_actuals = scaler.transform(np.concatenate(actuals))

print(scaled_actuals)

# Calculate Mean Squared Error (MSE) on scaled values
mse = mean_squared_error(scaled_actuals, scaled_predictions)
print("Overall MSE (scaled):", mse)
