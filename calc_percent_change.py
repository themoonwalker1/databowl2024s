import pandas as pd
import numpy as np

# Load the CSV file
data = pd.read_csv('data/data.csv')

# Select columns for which you want to calculate percent changes
columns_to_calculate = ['Air_Pollution_Deaths', 'Infectious_Disease_Deaths', 'Life_Expectancy',
                        'Smoking_Attributed_Deaths', 'Under_Fifteen_Mortality', 'Ozone_Pollution_Deaths',
                        'Health_Expenditure_as_GDP_Percentage']

# Group by 'Country' and calculate percent changes for selected columns
grouped = data.groupby('Country')[columns_to_calculate]
percent_changes = grouped.pct_change() * 100

# Concatenate percent changes with 'Country' and 'Year' columns
result = pd.concat([data[['Country', 'Year']], percent_changes], axis=1)
# Drop rows with NaN, infinite or undefined values
result = result.replace([np.inf, -np.inf], np.nan).dropna()
print(result)

# Save the result to a new CSV file
result.to_csv('percent.csv', index=False)
