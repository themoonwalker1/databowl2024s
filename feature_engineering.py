import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv("data/percent.csv")

# Display basic information about the dataset
print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Group by country
grouped_df = df.groupby('Country')

# Visualize correlations and pairplots for each country
for country, data in grouped_df:
    # Visualizing correlations between numerical features for each country
    correlation_matrix = data.drop(columns=['Country', 'Year']).corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"Correlation Matrix for {country}")
    plt.show()

    # Pairplot to visualize relationships between numerical features and target variable for each country
    sns.pairplot(data, vars=["Air_Pollution_Deaths", "Infectious_Disease_Deaths", "Life_Expectancy",
                             "Smoking_Attributed_Deaths", "Under_Fifteen_Mortality", "Ozone_Pollution_Deaths",
                             "Health_Expenditure_as_GDP_Percentage"])
    plt.suptitle(f"Pairplot for {country}")
    plt.show()
