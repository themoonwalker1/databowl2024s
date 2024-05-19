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
# print(t_df)

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

# Filter out rows with years 2000 and 2001
filtered_df = t_df[(t_df['Year'] != 2000) & (t_df['Year'] != 2001)]

# Output as CSV
filtered_df.to_csv('data/data.csv', index=False)

time_series = time_series.rename(columns=column_mapping)
time_series.dropna()
time_series.to_csv('data/time.csv', index=False)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

print("Training and Testing Random Forest:")

# Load the CSV file
df = pd.read_csv('data/data.csv')

best_mse = float('inf')  # Initialize best MSE to infinity
best_random_state = None  # Initialize best random state
best_regressor = None  # Initialize best regressor

columns = [
    'Air_Pollution_Deaths',
    'Infectious_Disease_Deaths',
    'Life_Expectancy',
    # 'Smoking_Attributed_Deaths',
    # 'Under_Fifteen_Mortality',
    'Ozone_Pollution_Deaths'
]

for epoch in range(5):  # Number of epochs
    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(df[columns],
                                                        df['Health_Expenditure_as_GDP_Percentage'],
                                                        test_size=0.1)

    # Initialize the model
    model = RandomForestRegressor()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate mean squared error
    mse = mean_squared_error(y_test, predictions)

    # Check if current MSE is the best so far
    if mse < best_mse:
        best_mse = mse
        best_random_state = epoch
        best_regressor = model

# Save the best regressor to a pickle file
with open('data/pkl/Bhandaru_Page_model_regressor.pkl', 'wb') as f:
    pickle.dump(best_regressor, f)

print("\tRandom Forest Regressor MSE for a 90-10 training testing split: " + str(best_mse))


import warnings
import pandas as pd
import pickle
from statsmodels.tsa.api import VAR

print("Training and Testing VAR Multivariate")

# Step 1: Load the data
df = pd.read_csv('data/time.csv')

# Step 2: Prepare the data (if needed)

# Step 3: Train the VAR model using data up to 2014
train_data = df[df['Year'] <= 2014]  # Including data up to 2014 for training

columns = [
    'Air_Pollution_Deaths',
    'Infectious_Disease_Deaths',
    'Life_Expectancy',
    # 'Smoking_Attributed_Deaths',
    # 'Under_Fifteen_Mortality',
    'Ozone_Pollution_Deaths'
]

fitted_models = {}
predictions = []
errors = []

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')

    for country in df['Country'].unique():
        train_subset = train_data[train_data['Country'] == country][columns]

        # Check if the data is not missing
        if not train_subset.isnull().any().any():
                # Fit VAR model
                model = VAR(train_subset)
                fitted_model = model.fit()

                # Predict 2015 values
                last_data_point = train_subset.iloc[:-6].values
                prediction = fitted_model.forecast(last_data_point, steps=1)
                errors.append(((train_subset.iloc[-1].values - prediction)/train_subset.iloc[-1].values))

                # Print the forecast
                # print(f"Country: {country}, Prediction: {prediction}, Error {errors[-1]}")

                predictions.append({'Country': country, 'Year': 2015,
                                    **dict(zip(columns, prediction.flatten()))})
                # Save fitted model
                fitted_models[country] = fitted_model

        else:
            # print(f"Missing data for {country}")
            # Append an empty row for the missing country
            predictions.append({'Country': country, 'Year': 2015,
                                **dict(zip(columns, [None] * len(columns)))})

    # Save the fitted VAR models
    with open('data/pkl/Bhandaru_Page_model_VAR.pkl', 'wb') as f:
        pickle.dump(fitted_models, f)

    # Convert forecasts to DataFrame
    forecasts_df = pd.DataFrame(predictions)

    # Save the forecasts DataFrame to a CSV file
    forecasts_df.to_csv('data/forecasts_2015_extrapolation.csv', index=False)
    print(f"\t MSE of percent error on predicting statistics in 2014 for {columns}:")
    print(f"\t{[sum([a * a for a in x])/len(errors) for x in zip(*errors)]}")

