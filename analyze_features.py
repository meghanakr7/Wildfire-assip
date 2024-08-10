import os
import zipfile
import numpy as np
import pandas as pd
from pytorch_tabnet.tab_model import TabNetRegressor

# Step 1: Unzip the model
model_zip_path = 'best_tabnet_model.zip'
model_dir = './'

with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
    zip_ref.extractall(model_dir)

# Step 2: Load the model
model_path = os.path.join(model_dir, 'best_tabnet_model.zip')  # Adjust if necessary
loaded_model = TabNetRegressor()
loaded_model.load_model(model_path)

# Step 3: Make predictions and analyze feature importance
# Load your data
data_path = 'wildfire.csv'
data = pd.read_csv(data_path)

# Assuming you have a feature matrix `X` and labels `y`
X = data.drop(columns=['FRP'])  # Replace 'FRP' with your actual target column
y = data['FRP']

# Predict
preds = loaded_model.predict(X.values)

# Feature importance
explain_matrix, masks = loaded_model.explain(X.values)
feature_importances = explain_matrix.mean(axis=0)
feature_names = X.columns

# Print feature importances
print("Feature Importances:")
for name, importance in zip(feature_names, feature_importances):
    print(f"{name}: {importance}")

# Save the feature importances to a file
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
})
feature_importance_df.to_csv('feature_importances.csv', index=False)

print("Feature importances have been saved to feature_importances.csv")