import pandas as pd

# Load the CSV file
df = pd.read_csv('/Users/meghana/Documents/projects/AutoKeras/data/extracted_rows_20210701.csv')

# Keep only the first 17 rows
df = df.iloc[:17].reset_index(drop=True)

# Define the new values (numerical only)
new_values = [
    26.17, 20, 14247.34, 50.71, 211851,
    4430.84, 2.5, 45.45, 22.99, 3.5,
    0, 50, 10, 1, 1, 14.8, 6.45
]

# Add a new column with the predicted values
df['GPT predicted value'] = new_values

# Save the updated CSV file
df.to_csv('/Users/meghana/Documents/projects/AutoKeras/data/updated_extracted_rows_20210701.csv', index=False)
