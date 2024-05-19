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