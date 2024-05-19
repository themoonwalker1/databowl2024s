import pandas as pd
import numpy as np

filtered_df = pd.read_csv('data/percent.csv')
y_axis = 'Health_Expenditure_as_GDP_Percentage'

# Group by the first column and select only the second and ninth columns
linear_regression_data = filtered_df[['Country', 'Year', y_axis]]

import matplotlib.pyplot as plt

# Assuming linear_regression_data is your DataFrame
# Replace 'linear_regression_data' with the actual name of your DataFrame if it's different

# Group the data by 'Entity'
linear_groups = linear_regression_data.groupby('Country')

# Calculate the mean expenditure for each entity and sort the groups accordingly
sorted_groups = linear_groups.mean().sort_values(by=y_axis, ascending=True)

# Get the unique entities
entities = sorted_groups.index

# Set the number of entities per plot
entities_per_plot = 20

# Calculate the total number of plots needed
num_plots = len(entities) // entities_per_plot
if len(entities) % entities_per_plot != 0:
    num_plots += 1

# Create subplots
fig, axs = plt.subplots(num_plots, figsize=(12, 8 * num_plots))

# Iterate through each plot
for i, ax in enumerate(axs):
    start_idx = i * entities_per_plot
    end_idx = min((i + 1) * entities_per_plot, len(entities))

    # Iterate through each entity for the current plot
    for entity in entities[start_idx:end_idx]:
        group_data = linear_groups.get_group(entity)
        ax.plot(group_data['Year'], group_data[
            y_axis],
                marker='o', linestyle='-', label=entity)

    # Set plot title and labels
    ax.set_title(
        'Health Expenditure as Percentage of GDP Over Time (Entities {})'.format(
            start_idx + 1))
    ax.set_xlabel('Year')
    ax.set_ylabel('Percentage of GDP')

    # Show legend
    ax.legend()

    # Show grid
    ax.grid(True)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()
