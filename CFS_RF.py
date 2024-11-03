import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def correlation_feature_selection(X, y, k=10):
    """
    Correlation Feature Selection (CFS) algorithm.

    Parameters:
    X (DataFrame): Features.
    y (Series): Target variable.
    k (int): Number of features to select.

    Returns:
    selected_features (list): List of selected features.
    """

    # Filter out non-numeric columns
    numeric_columns = X.select_dtypes(include=np.number).columns
    X_numeric = X[numeric_columns]

    # Calculate correlation matrix
    corr_matrix = X_numeric.corr()

    # Initialize feature scores
    feature_scores = []

    # Iterate through features
    for feature in X_numeric.columns:
        # Calculate correlation between feature and target
        correlation = abs(X_numeric[feature].corr(y))
        # Calculate average correlation with other features
        avg_corr_with_others = (np.sum(abs(corr_matrix[feature])) - 1) / (len(X_numeric.columns) - 1)
        # Calculate feature score
        feature_score = correlation / avg_corr_with_others
        feature_scores.append((feature, feature_score))

    # Sort feature scores in descending order
    feature_scores.sort(key=lambda x: x[1], reverse=True)

    # Select top k features
    selected_features = [x[0] for x in feature_scores[:k]]

    return selected_features

data = pd.read_csv('D:\Downloads\Autism\Autism\Autism\Autism-20240313T052123Z-001\Autism.csv')

# Assume 'target' is the name of the target variable column
X = data.drop(columns=['ASD TRAITS'])
y = data['ASD TRAITS']

selected_features = correlation_feature_selection(X, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use selected features for training and testing
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Train the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_selected, y_train)

# Predictions
y_pred = rf_classifier.predict(X_test_selected)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Selected Features:", selected_features)
print("Accuracy:",accuracy)


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Get predicted probabilities for the positive class
y_pred_proba = rf_classifier.predict_proba(X_test_selected)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Calculate AUC (Area Under the Curve)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

import pickle

# Save the trained Random Forest classifier
with open('D:/Downloads/Autism/Autism/Autism/Autism-20240313T052123Z-001/random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_classifier, f)

# Save the selected features
with open('selected_features.txt', 'w') as f:
    for feature in selected_features:
        f.write("%s\n" % feature)

print("Model and selected features saved successfully.")

# %%



