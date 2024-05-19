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

import pandas as pd

# Load the predicted DataFrame
predicted_df = pd.read_csv('data/predicted.csv')

# Filter data for the year 2014 and 2015
year_2014 = predicted_df[predicted_df['Year'] == 2014]
year_2015 = predicted_df[predicted_df['Year'] == 2015]

# Merge data for 2014 and 2015 based on country
merged_df = pd.merge(year_2014, year_2015, on='Country', suffixes=('_2014', '_2015'))

# Calculate percent increase
merged_df['Percent_Increase'] = ((merged_df['Health_Expenditure_as_GDP_Percentage_2015'] - merged_df['Health_Expenditure_as_GDP_Percentage_2014']) / merged_df['Health_Expenditure_as_GDP_Percentage_2014']) * 100
# print(merged_df['Percent_Increase'])
# Find the country with the largest increase
merged_df.sort_values(by='Percent_Increase', ascending=False, inplace=True)

pd.options.display.max_columns = None
print("\n\tTop 3 countries with the largest increase in this trial:")
print(merged_df.head(3))


