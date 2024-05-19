import pandas as pd
import pickle

# Load the trained regressor
with open('data/pkl/Bhandaru_Page_model_regressor.pkl', 'rb') as f:
    regressor = pickle.load(f)

# Load the forecasts for 2015 extrapolation
forecasts = pd.read_csv('data/forecasts_2015_extrapolation.csv')
df = pd.read_csv('data/data.csv')

# Initialize a list to store the predicted health expenditures
predicted_values = []

# Iterate over each country's forecast
for index, row in forecasts.iterrows():
    # Extract features for prediction
    features = row.drop(['Country', 'Year']).values.reshape(1, -1)

    # Predict health expenditure for 2015
    predicted_expenditure = regressor.predict(features)[0]

    # Append the predicted value to the list
    predicted_values.append({'Country': row['Country'], 'Year': 2015, 'Health_Expenditure_as_GDP_Percentage': predicted_expenditure})

# Create a DataFrame from the predicted values
predicted_df = pd.DataFrame(predicted_values)
predicted_df = pd.concat([predicted_df, df[['Country', 'Year', 'Health_Expenditure_as_GDP_Percentage']]])
predicted_df.sort_values(by='Year', ascending=True, inplace=True)
predicted_df.sort_values(by='Country', ascending=True, inplace=True)
# Save the predicted values to a CSV file
predicted_df.to_csv('data/predicted.csv', index=False)



