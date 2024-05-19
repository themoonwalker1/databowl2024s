import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Load data
datasets = [
    ('data/original/air pollution deaths - Sheet1.csv', 'Deaths_that_are_from_all_causes_attributed_to_air_pollution_in_both_sexes_aged_all_ages'),
    ('data/original/immunization of hepb3 of one year old children - Sheet1.csv', 'HepB3__of_oneyearolds_immunized'),
    ('data/original/infectious diseases deaths - Sheet1.csv', 'Deaths_from_Infectious_diseases'),
    ('data/original/life expectancy - Sheet1.csv', 'Life_expectancy_at_birth'),
    ('data/original/number of deaths from tobacco smoking - Sheet1.csv', 'Deaths_that_are_from_all_causes_attributed_to_smoking'),
    ('data/original/number of youth deaths - Sheet1.csv', 'Under_fifteen_mortality__Number_of_deaths'),
    ('data/original/ozone pollution deaths - Sheet1.csv', 'Deaths_that_are_from_all_causes_attributed_to_ambient_ozone_pollution'),
    ('data/original/share of women contraception - Sheet1.csv', 'Proportion_of_women_of_reproductive_age_aged_1549_years_who_have_their_need_for_family_planning_satisfied_with_modern_methods__of_women_aged_1549_years'),
    ('data/original/total healthcare expenditure as share of gdp - Sheet1.csv', 'Current_health_expenditure_CHE_as_percentage_of_gross_domestic_product_GDP_'),
    ('data/original/universal health coverage index - Sheet1.csv', 'UHC_Index')
]

dfs = []
identifiers = []

for dataset, column_name in datasets:
    data = np.genfromtxt(dataset, dtype=None, delimiter=',', names=True, usecols=(0,2,3))
    identifiers.append(dataset.split(' - ')[0])
    df = pd.DataFrame(data)
    df = df.rename(columns={df.columns[2]: column_name})  # Rename the 3rd column

    dfs.append(df)

# Merge DataFrames
t_df = dfs[0]
for df in dfs[1:]:
    t_df = pd.merge(t_df, df, how='left', on=['Entity', 'Year'])

# Display the result
print(t_df)

t_df = t_df.drop(columns=['UHC_Index', 'Proportion_of_women_of_reproductive_age_aged_1549_years_who_have_their_need_for_family_planning_satisfied_with_modern_methods__of_women_aged_1549_years', 'HepB3__of_oneyearolds_immunized'])
time_series = t_df.drop(columns=['Current_health_expenditure_CHE_as_percentage_of_gross_domestic_product_GDP_'])
t_df = t_df.dropna()

# Original column names
original_columns = [
    'Entity',
    'Year',
    'Deaths_that_are_from_all_causes_attributed_to_air_pollution_in_both_sexes_aged_all_ages',
    'Deaths_from_Infectious_diseases',
    'Life_expectancy_at_birth',
    'Deaths_that_are_from_all_causes_attributed_to_smoking',
    'Under_fifteen_mortality__Number_of_deaths',
    'Deaths_that_are_from_all_causes_attributed_to_ambient_ozone_pollution',
    'Current_health_expenditure_CHE_as_percentage_of_gross_domestic_product_GDP_'
]

# New column names
new_columns = [
    'Country',
    'Year',
    'Air_Pollution_Deaths',
    'Infectious_Disease_Deaths',
    'Life_Expectancy',
    'Smoking_Attributed_Deaths',
    'Under_Fifteen_Mortality',
    'Ozone_Pollution_Deaths',
    'Health_Expenditure_as_GDP_Percentage'
]

# Map original column names to new column names
column_mapping = dict(zip(original_columns, new_columns))

t_df.rename(columns=column_mapping, inplace=True)

print(t_df)

# Filter out rows with years 2000 and 2001
filtered_df = t_df[(t_df['Year'] != 2000) & (t_df['Year'] != 2001)]

# Output as CSV
filtered_df.to_csv('data/data.csv', index=False)

time_series = time_series.rename(columns=column_mapping)
time_series.dropna()
time_series.to_csv('data/time.csv', index=False)



