# Import necessary libraries
from sklearn.linear_model import LinearRegression
import pandas as pd

# Create dataset with housing features and prices
data = pd.DataFrame({
    'area': [50, 80, 100, 120, 150, 70, 90, 110],          # Square meters
    'construction_year': [1390, 1395, 1400, 1385, 1398, 1392, 1396, 1399],  # Year built
    'floor': [2, 3, 4, 2, 5, 3, 3, 4],                     # Floor number
    'rooms': [1, 2, 3, 3, 4, 2, 2, 3],                     # Number of rooms
    'elevator': [0, 1, 1, 0, 1, 0, 1, 1],                  # Elevator (0=No, 1=Yes)
    'parking': [1, 1, 1, 0, 1, 0, 1, 1],                   # Parking (0=No, 1=Yes)
    'warehouse': [0, 1, 1, 1, 1, 0, 0, 1],                 # Warehouse (0=No, 1=Yes)
    'renovation': [1, 0, 0, 2, 1, 1, 0, 0],                # Renovation level (0-2)
    'price': [500, 800, 1000, 1200, 1500, 650, 900, 1100]   # Price in million units
})

# Define features (X) and target variable (y)
X = data[['area', 'construction_year', 'floor', 'rooms', 'elevator', 'parking', 'warehouse', 'renovation']]
y = data['price']

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Make a prediction for a new house
# Features: area=90, construction_year=1397, floor=3, rooms=2, elevator=1, parking=1, warehouse=0, renovation=0
new_house = [[90, 1397, 3, 2, 1, 1, 0, 0]]
predicted_price = model.predict(new_house)

# Display the predicted price
print(f"Predicted Price: {int(predicted_price[0])} million units")
