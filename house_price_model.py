#1. Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#2. Loading Dataset
data = pd.read_csv('data_house.csv')
print(data.head())

#3. Data Cleaning

# Remove rows with missing values
data = data.dropna()

# Remove duplicate rows
data = data.drop_duplicates()

# Display the number of remaining rows and columns to confirm cleaning
print(f'Dataset after cleaning: {data.shape}')

#4. Feature Selection

# Define the target variable and features
X = data[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'grade', 'sqft_above', 'sqft_basement']]
y = data['price']

#Spliting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#5. model Selection
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#6. Model Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-Squared; {r2}')

#7. Visualization
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actural vs. Predicted Prices')
plt.show()

residuals = y_test - y_pred
sns.histplot(residuals, kde=True)
plt.title('Residuals Distribution')
plt.show()

#Saving and loading the model using pickle
import pickle

#save the model
with open('linear_model.pkl', 'wb') as file:
    pickle.dump(model, file)