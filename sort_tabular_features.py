import pandas as pd
from tabulate import tabulate

# Read the CSV file
df = pd.read_csv('feature_importances.csv')

# Sort the DataFrame based on the 'importance' column (replace 'importance' with the actual column name if different)
df_sorted = df.sort_values(by='Importance', ascending=False)

# Save the sorted DataFrame to a clear tabular format in a text file
with open('sorted_feature_importance_table.txt', 'w') as f:
    f.write(tabulate(df_sorted, headers='keys', tablefmt='grid'))

# Optionally, display the sorted DataFrame as a table
print(tabulate(df_sorted, headers='keys', tablefmt='grid'))