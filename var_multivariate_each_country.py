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

