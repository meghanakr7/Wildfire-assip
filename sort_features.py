import pandas as pd

# Read the CSV file
df = pd.read_csv('feature_importances.csv')

# Sort the DataFrame based on the 'importance' column (replace 'importance' with the actual column name if different)
df_sorted = df.sort_values(by='Importance', ascending=False)

# Save the sorted DataFrame to a new CSV file
df_sorted.to_csv('sorted_feature_importance.csv', index=False)

# Optionally, display the sorted DataFrame
print(df_sorted)