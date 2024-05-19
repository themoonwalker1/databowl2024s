import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import warnings

# Step 1: Load the data
df = pd.read_csv('data/time.csv')

# Step 2: Prepare the data (if needed)

# Step 3: Split the data
train_data = df[df['Year'] < 2014]
test_data = df[df['Year'] == 2014]

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    # Step 4: Model fitting and prediction
    for statistic in ['Air_Pollution_Deaths', 'Infectious_Disease_Deaths',
                      'Life_Expectancy',
                      'Smoking_Attributed_Deaths', 'Under_Fifteen_Mortality',
                      'Ozone_Pollution_Deaths']:

        predictions = []
        actuals = []
        for country in df['Country'].unique():
            train_subset = train_data[train_data['Country'] == country][statistic]
            test_subset = test_data[test_data['Country'] == country][statistic]

            # Check if the data is not missing
            if not train_subset.isnull().any() and not test_subset.isnull().any():
                # Fit ARMA model
                try:
                    model = ARIMA(train_subset,
                                  order=(5, 0, 5))  # Example order, adjust as needed
                    fitted_model = model.fit()

                    # Predict 2015 values
                    prediction = fitted_model.forecast(steps=len(test_subset))

                    predictions.append(prediction.tolist()[0])
                    actuals.append(test_subset.tolist()[0])

                    # Evaluate prediction
                    # You can calculate metrics such as MAE, MSE, RMSE here
                    print(
                        f"Country: {country}, Statistic: {statistic}, Prediction: {prediction.tolist()[0]}, Actual {test_subset.tolist()[0]}, Percent Difference {((test_subset.tolist()[0]-prediction.tolist()[0])/test_subset.tolist()[0]) * 100}%")


                except Exception as e:
                    print(f"Error occurred for {country}, {statistic}: {e}")

            else:
                print(f"Missing data for {country}, {statistic}")

        print(mean_squared_error(actuals, predictions))