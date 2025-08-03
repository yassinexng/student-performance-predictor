import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score  
import warnings
warnings.filterwarnings('ignore')

# Dataset Loading
file_path = "/home/yassine/.cache/kagglehub/datasets/rabieelkharoua/students-performance-dataset/versions/2/Student_performance_data .csv"

try:
    data = pd.read_csv(file_path)
except Exception as e: 
   print("An error was found while opening the dataset: %s" % str(e)) 
   sys.exit()

print("\nFirst 5 rows of the dataset:") 
print(data.head())
print("\nLast 5 rows of the dataset:") 
print(data.tail())

print("\nDataset Info:")
data.info() 

print("\nDataset Description (Numerical Columns):")
print(data.describe())

print("Number of missing values per column:")
print(data.isnull().sum())

duplicates = data.duplicated().sum()
print("\nNumber of duplicate rows: %s" % str(duplicates)) 

if duplicates > 0:
    data.drop_duplicates(inplace=True)
    print("Duplicates were dropped successfully. New shape: %s" % str(data.shape))
else:
    print("No duplicate rows found in the dataset.")

# Univariate Analysis - Histograms
numerical_features = ['StudyTimeWeekly', 'Absences', 'ParentalEducation', 'GPA']
size = len(numerical_features)

plt.figure(figsize=(8, size * 4))

i = 0
for feature in numerical_features:
    plt.subplot(size, 1, i + 1)
    plt.hist(data[feature], bins=20, edgecolor='black')
    plt.title('Distribution of %s' % feature)
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    i += 1

plt.tight_layout()
plt.show()

# Univariate Analysis - Bar Charts
categorical_features = ['Extracurricular', 'ParentalEducation']

plt.figure(figsize=(14, 7))
i = 0
for feature in categorical_features:
    plt.subplot(1, len(categorical_features), i + 1)

    counts = data[feature].value_counts()
    categories = counts.index
    values = counts.values
    
    
    plt.bar(categories, values, color='blue') 
    plt.title('Count of %s' % feature)
    plt.xlabel(feature)
    plt.ylabel('Count')
    
    i += 1

plt.tight_layout()
plt.show()

print("\nValue counts for categorical features:")
for feature in categorical_features:
    print("\n%s: " % feature)
    counts = data[feature].value_counts()
    total = len(data[feature])
    
    print(counts)
    for category, count in counts.items():
        percentage = (float(count) / total) * 100
        print("    %s: %s%%" % (str(category), str(round(percentage, 2))))

# Data Cleaning
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.fillna(data.mean(numeric_only=True), inplace=True) 
print("\nMissing and infinite numerical values handled successfully.")
print("\nNumber of missing values after handling:")
print(data.isnull().sum())

# The custom StandardScaler class that we made:
class MyStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
    
    def fit(self, X):
        X = np.asarray(X)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
    
    def transform(self, X):
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("The scaler has not been fitted yet.")
        return (X - self.mean_) / self.scale_
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_scaled):
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("The scaler has not been fitted yet.")
        return (X_scaled * self.scale_) + self.mean_

# Data Preparation:

selected_features = ['StudyTimeWeekly', 'Absences', 'ParentalEducation', 'Extracurricular']
X_raw = data[selected_features]
Y = data['GPA'].values

# We then split the data into training and testing sets.
X_train_raw, X_test_raw, Y_train, Y_test = train_test_split(
    X_raw, Y, test_size=0.2, random_state=42
)
print("\nData has been split into training and testing sets:")
print("Training features (X_train) shape: %s" % str(X_train_raw.shape))
print("Testing features (X_test) shape: %s" % str(X_test_raw.shape))
print("Training target (Y_train) shape: %s" % str(Y_train.shape))
print("Testing target (Y_test) shape: %s" % str(Y_test.shape))


# We initialize the scaler:
scaler = MyStandardScaler()

# We fit the scaler ONLY on the training data, then transform the training data.
X_train_scaled = scaler.fit_transform(X_train_raw)
print("\nTraining features scaled using MyStandardScaler.")
print("Scaled training features (X_train_scaled) shape: %s" % str(X_train_scaled.shape))

# Transform the test data using the mean and scale learned from the training data.
X_test_scaled = scaler.transform(X_test_raw)
print("Testing features scaled using the same MyStandardScaler.")
print("Scaled testing features (X_test_scaled) shape: %s" % str(X_test_scaled.shape))
print("\nData preprocessing complete. Data is now ready for model building.")


# The Custom Linear Regression Functions:

def predict(X, w, b):
    return np.dot(X, w) + b

def cost_function(X, y, w, b):
    m = len(y)
    predictions = predict(X, w, b)
    cost = (1.0 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

def gradient_descent(X, y, learning_rate, iterations):
    m, n = X.shape
    w = np.zeros(n)
    b = 0.0
    cost_history = []

    for i in range(iterations):
        predictions = predict(X, w, b)
        errors = predictions - y
        
        dw = (1.0 / m) * np.dot(X.T, errors)
        db = (1.0 / m) * np.sum(errors)
        
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        cost = cost_function(X, y, w, b)
        cost_history.append(cost)

    return w, b, cost_history

# Gradient Descent Training:
learning_rate = 0.01
iterations = 1000

print("\nPhase 1: Gradient Descent initiated for model training:")


w_learned, b_learned, cost_history = gradient_descent(X_train_scaled, Y_train, learning_rate, iterations)

print("\nModel training complete.")
print("Learned Weights (w): %s" % str(w_learned))
print("Learned Bias (b): %s" % str(b_learned))


# Cost History:
print("\nCost history progression:")
for i in range(len(cost_history)):
    if (i % 100 == 0) or (i == len(cost_history) - 1):
        print("At iteration No %d, the cost was: %s" % (i, str(cost_history[i])))

# Model Evaluation
print("\nModel Evaluation:")
y_train_pred = predict(X_train_scaled, w_learned, b_learned)
y_test_pred = predict(X_test_scaled, w_learned, b_learned)

train_mse = mean_squared_error(Y_train, y_train_pred)
train_r2 = r2_score(Y_train, y_train_pred)

test_mse = mean_squared_error(Y_test, y_test_pred)
test_r2 = r2_score(Y_test, y_test_pred)

print("Training set MSE: %s" % str(round(train_mse, 4)))
print("Testing set MSE: %s" % str(round(test_mse, 4)))
print("Training set R^2 Score: %s" % str(round(train_r2, 4)))
print("Testing set R^2 Score: %s" % str(round(test_r2, 4)))

# Prediction Clipping
y_test_pred_clipped = np.clip(y_test_pred, 0.0, 4.0)

clipped_mse = mean_squared_error(Y_test, y_test_pred_clipped)
clipped_r2 = r2_score(Y_test, y_test_pred_clipped)

print("Test MSE after clipping predictions: %s" % str(round(clipped_mse, 4)))
print("Test R^2 after clipping predictions: %s" % str(round(clipped_r2, 4)))

# Visualizing Predictions
print("\nVisualizing Predictions:")
plt.figure(figsize=(10, 6))
plt.scatter(Y_test, y_test_pred_clipped, alpha=0.7)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--', lw=2)
plt.title('Actual vs. Predicted GPA (Test Set)')
plt.xlabel('Actual GPA')
plt.ylabel('Predicted GPA')
plt.grid(True)
plt.show()
