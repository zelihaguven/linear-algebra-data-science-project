 #Linear_Algebra_Data_Science_Project.py


# Import necessary libraries

pip install numpy pandas scikit-learn


import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Step 1: Data Selection
# Load the diabetes dataset
diabetes_data = load_diabetes()
X = diabetes_data.data
y = diabetes_data.target

# Step 2: Data Exploration and Cleaning
# Convert to DataFrame for easier analysis
df = pd.DataFrame(data=X, columns=diabetes_data.feature_names)
df['target'] = y

# Check for missing values
print("Missing values in the dataset:\n", df.isnull().sum())

# Step 3: PCA for Dimensionality Reduction
# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=5)  # Reduce to 5 components
X_pca = pca.fit_transform(X_scaled)

# Step 4: Linear Regression Model using the reduced dataset
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Evaluate Model Performance
# Make predictions
y_pred = model.predict(X_test)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the results
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R^2 Score: {r2}")


