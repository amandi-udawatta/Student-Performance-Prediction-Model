import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('Student_Performance.csv')

# Convert categorical data to numerical
data['Extracurricular Activities'] = data['Extracurricular Activities'].apply(lambda x: 1 if x == 'Yes' else 0)

#get the features and labels from the dataset  
features = data[['Hours Studied' , 'Previous Scores' , 'Extracurricular Activities' , 'Sleep Hours' , 'Sample Question Papers Practiced']]
target = data['Performance Index'].values.reshape(-1, 1)
    
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


new_data = np.array([[5, 77, 0, 8, 2]])  # Modify the example input, ensuring categorical data is numeric
new_data_normalized = (new_data - features_mean.values) / features_std.values  # Ensure correct broadcasting
new_data_b = np.c_[np.ones((1, 1)), new_data_normalized]


prediction = new_data_b.dot(theta)
print("Prediction for new data:", prediction)


# Create subplots to visualize the effect of each feature
fig, axes = plt.subplots(3, 2, figsize=(15, 10))

features_list = ['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']

for i, ax in enumerate(axes.flatten()[:-1]):  # Leave the last subplot empty for neatness
    ax.scatter(features[features_list[i]], target, color='blue', label='Training data')
    
    # Predict values for the regression line
    feature_values = np.linspace(features[features_list[i]].min(), features[features_list[i]].max(), 100).reshape(-1, 1)
    feature_values_normalized = (feature_values - features_mean[i]) / features_std[i]
    
    X_feature_b = np.c_[np.ones((feature_values_normalized.shape[0], 1)), np.zeros((feature_values_normalized.shape[0], features_normalized.shape[1]))]
    X_feature_b[:, i+1] = feature_values_normalized.flatten()
    
    target_predictions = X_feature_b.dot(theta)
    
    ax.plot(feature_values, target_predictions, color='red', label='Regression line')
    ax.set_xlabel(features_list[i])
    ax.set_ylabel("Performance Index")
    ax.legend()

# Remove the last subplot (for neatness)
fig.delaxes(axes.flatten()[-1])

fig.suptitle("Effect of Each Feature on Performance Index")
plt.tight_layout()
plt.show()