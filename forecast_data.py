import pandas as pd
import pickle

# Step 1: Load the data
df = pd.read_csv('data/time.csv')

# Step 2: Prepare the data (if needed)

# Load the fitted VAR models from the pickle file
with open('fitted_var_models.pkl', 'rb') as f:
    fitted_models = pickle.load(f)

# Step 3: Forecast 2015 values for each country
all_statistics = ['Air_Pollution_Deaths', 'Infectious_Disease_Deaths',
                  'Life_Expectancy',
                  'Smoking_Attributed_Deaths', 'Under_Fifteen_Mortality',
                  'Ozone_Pollution_Deaths']

forecasts = []

for country, model in fitted_models.items():
    test_subset = df[(df['Country'] == country) & (df['Year'] == 2014)][all_statistics]
    if not test_subset.empty:
        try:
            # Forecast 2015 values
            forecast = model.forecast(test_subset.values, steps=1).flatten()

            # Append forecast to forecasts list
            forecasts.append({'Country': country, 'Year': 2015, **dict(zip(all_statistics, forecast))})

        except Exception as e:
            print(f"Error occurred for {country}: {e}")

    else:
        print(f"Missing data for {country}")

# Convert forecasts to DataFrame
forecasts_df = pd.DataFrame(forecasts)

# Save the forecasts DataFrame to a CSV file
forecasts_df.to_csv('data/forecasts.csv', index=False)
