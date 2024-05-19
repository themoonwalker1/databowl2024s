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
