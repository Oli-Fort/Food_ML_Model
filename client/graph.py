import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load the dataset from the specified path
dataset_path = os.path.join(os.path.dirname(__file__), 'foodXclusters.csv')
dataset = pd.read_csv(dataset_path)

# Extract features (X) and target labels (y) from the dataset
X = dataset.iloc[:, 0].values  # Assuming the first column is the feature
y = dataset.iloc[:, 1].values  # Assuming the second column is the target label

# Encode string labels into integers
le = LabelEncoder()
X_encoded = le.fit_transform(X)  # Fit label encoder and return encoded labels
# Printing the classes can help understand the encoding mapping
print(list(le.classes_))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.25, random_state=0)

# Initialize and train a RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
# Note: X_train and X_test need to be reshaped for a single feature input
classifier.fit(X_train.reshape(-1, 1), y_train)

# Predict the target values for the test set
y_pred = classifier.predict(X_test.reshape(-1, 1))

# Print a comparison between predicted and actual values
# np.concatenate is used to merge the predicted and actual values side by side for easy comparison
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), axis=1))
