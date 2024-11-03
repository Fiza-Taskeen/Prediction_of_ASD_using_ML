import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report

# Load your dataset (replace 'your_dataset.csv' with your actual file)
data = pd.read_csv('D:\Downloads\Autism\Autism\Autism\Autism-20240313T052123Z-001\Autism.csv')

# Extract features and labels
X = data.drop(columns=['ASD TRAITS'])  # Assuming 'label' is the column containing the target variable
y = data['ASD TRAITS']


X = pd.get_dummies(X,columns = ['ETHNICITY','WHO COMPLETED THE TEST'])
# Convert the target variable to binary (0 or 1)
y_binary = (y > y.mean()).astype(int)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the linear regression classifier
linear_reg_model = LinearRegression()
linear_reg_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = linear_reg_model.predict(X_test_scaled)

# Convert predicted probabilities to binary outcomes (0 or 1) using a threshold
threshold = 0.7# Adjust as needed
y_pred_binary = (y_pred > threshold).astype(int)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_binary)
print(f'Accuracy: {accuracy:.4f}')

# Display classification report for more detailed metrics
print("Classification Report:\n", classification_report(y_test, y_pred_binary))

import matplotlib.pyplot as plt

# Plotting the actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted Values')
plt.show()


# %%



