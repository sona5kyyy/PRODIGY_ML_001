# PRODIGY_ML_01 - House Price Prediction Model
# Overview
This project implements a Linear Regression model to predict house prices based on three features:

Square footage
Number of bedrooms
Number of bathrooms
The model is trained using a small sample dataset and can predict house prices for new input data. This project demonstrates how linear regression can be applied to real-world problems such as house price prediction.

# Features
The model uses the following features to predict house prices:

Square footage: The total area of the house in square feet.
Bedrooms: The number of bedrooms in the house.
Bathrooms: The number of bathrooms in the house.
# Dataset
The sample dataset consists of 15 entries, each with the square footage, number of bedrooms, number of bathrooms, and house price.

Here's a snippet of the dataset:

Square Footage	Bedrooms	Bathrooms	Price
1500	3	2	250000
2000	4	3	350000
1800	3	2	300000
...	...	...	...
The dataset can be created using the code provided below or saved manually.

How to Run the Project
1. Install Dependencies
Make sure you have Python installed, then install the required libraries using pip:

bash
Copy
pip install pandas scikit-learn
2. Save the Dataset
You can create and save the dataset using the following code:

python
Copy
import pandas as pd

# Create the dataset
data = {
    'square_footage': [1500, 2000, 1800, 2200, 1600, 2500, 1400, 3000, 1200, 2700, 1300, 3200, 2400, 2800, 2600],
    'bedrooms': [3, 4, 3, 4, 3, 4, 2, 5, 2, 4, 3, 5, 4, 4, 4],
    'bathrooms': [2, 3, 2, 3, 2, 3, 1, 4, 1, 3, 1, 4, 2, 3, 3],
    'price': [250000, 350000, 300000, 400000, 270000, 450000, 230000, 600000, 210000, 490000, 225000, 620000, 430000, 510000, 470000]
}

# Convert the dictionary to a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('house_prices.csv', index=False)
3. Run the Model
Once the dataset is saved, you can run the model using the following Python code:

python
Copy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('house_prices.csv')

# Split the data into features and target variable
X = data[['square_footage', 'bedrooms', 'bathrooms']]
y = data['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Display predictions vs. actual values
predicted_data = pd.DataFrame({
    'Predicted Price': y_pred,
    'Actual Price': y_test.values
})

print("Predicted vs Actual Prices for Test Data:")
print(predicted_data)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Predict new house price
def predict_price(square_footage, bedrooms, bathrooms):
    new_data = pd.DataFrame({
        'square_footage': [square_footage],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms]
    })

    # Normalize the new data
    new_data_scaled = scaler.transform(new_data)

    # Predict the price
    predicted_price = model.predict(new_data_scaled)
    return predicted_price[0]

# Example: Predict price for a house
new_square_footage = 1900
new_bedrooms = 3
new_bathrooms = 2

predicted_price = predict_price(new_square_footage, new_bedrooms, new_bathrooms)
print(f"\nPredicted Price for a house with {new_square_footage} sqft, {new_bedrooms} bedrooms, and {new_bathrooms} bathrooms: ${predicted_price:.2f}")
4. Predict New House Prices
To predict the price of a new house, you can use the predict_price function by passing values for square footage, bedrooms, and bathrooms.

For example:

python
Copy
predicted_price = predict_price(2000, 3, 2)
print(f"Predicted Price: ${predicted_price:.2f}")
5. Screenshot of the Output
Once you run the model, you should see output similar to the following:

Predicted vs Actual Prices for Test Data:
Predicted Price	Actual Price
280000.00	290000.00
400000.00	380000.00
...	...
Model Evaluation:
yaml
Copy
Mean Squared Error: 5500000000.00
R-squared Score: 0.89
Example Prediction:
javascript
Copy
Predicted Price for a house with 2000 sqft, 3 bedrooms, and 2 bathrooms: $350000.00
# Screenshot of the Output:

"D:\c drive data 22-12-2024\OneDrive\Pictures\Screenshots\Screenshot 2024-09-07 190813.png"

# Evaluation
After training the model, it is evaluated using the following metrics:

Mean Squared Error (MSE): Indicates the average of the squared errors between the predicted and actual prices.
R-squared Score: Measures the proportion of variance in the target variable (house price) that can be explained by the independent variables (square footage, bedrooms, bathrooms).
# Conclusion
This project demonstrates the application of Linear Regression for predicting house prices based on key features. The model can be further improved by adding more features such as location, age of the house, and other factors affecting price.
