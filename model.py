import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('Student_Performance.csv')

# Convert categorical data to numerical
data['Extracurricular Activities'] = data['Extracurricular Activities'].apply(lambda x: 1 if x == 'Yes' else 0)

#get the features and labels from the dataset  
features = data.drop('Performance Index', axis=1)
target = data['Performance Index'].values.reshape(-1, 1)

# Normalizing the features 
features_mean = features.mean(axis=0)
features_std = features.std(axis=0)
features_normalized = (features - features_mean) / features_std

# Prepare the design matrix
X_b = np.c_[np.ones((features_normalized.shape[0], 1)), features_normalized]

# Hyperparameters
alpha = 0.01
n_iterations = 10000
m = len(X_b)

# Initialize parameters close to zero
theta = np.zeros((X_b.shape[1], 1))

# Gradient Descent
for i in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - target)
    theta -= alpha * gradients

# Prepare new data for prediction (Data Transformation for New Data in Data Pipelining)
new_data = np.array([[5, 77, 0, 8, 2]])  # Modify the example input, ensuring categorical data is numeric
new_data_normalized = (new_data - features_mean.values.reshape(1, -1)) / features_std.values.reshape(1, -1)
new_data_b = np.c_[np.ones((1, 1)), new_data_normalized]

# Make a prediction with the new data (Model Inference Stage in Data Pipelining)
prediction = new_data_b.dot(theta)
print("Prediction for new data:", prediction)

# Predictions for training data
predictions = X_b.dot(theta)

# Residuals for training data
residuals = target - predictions

# Create subplots for residuals plot and predicted vs actual plot (1x2 grid)
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Residuals Plot (left plot)
axes[0].scatter(predictions, residuals, color='blue')
axes[0].axhline(y=0, color='red', linestyle='--')
axes[0].set_xlabel('Predicted Performance Index')
axes[0].set_ylabel('Residuals')
axes[0].set_title('Residuals vs Predicted Values')
axes[0].grid(True)

# Predicted vs Actual Performance Plot (right plot)
axes[1].scatter(target, predictions, color='green', label='Predicted vs Actual')
axes[1].plot([target.min(), target.max()], [target.min(), target.max()], 'k--', lw=2)  # Line of perfect prediction
axes[1].set_xlabel('Actual Performance Index')
axes[1].set_ylabel('Predicted Performance Index')
axes[1].set_title('Predicted vs Actual Performance')
axes[1].legend()
axes[1].grid(True)

# Adjust layout and show the plot
plt.tight_layout()
plt.subplots_adjust(wspace=0.4)  # Add space between subplots
plt.show()