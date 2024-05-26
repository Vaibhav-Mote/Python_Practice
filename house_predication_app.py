import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load data from Excel
data = pd.read_excel('house_data.xlsm', engine='openpyxl')

# Extract features and target
X = data[['SqFt', 'Bedrooms']]
y = data['Price']

# Train the model
model = LinearRegression()
model.fit(X, y)

# Plot the graph
fig, ax = plt.subplots()
ax.scatter(X['SqFt'], y, color='blue')
ax.plot(X['SqFt'], model.predict(X), color='red', linewidth=2)
ax.set_xlabel('Square Feet')
ax.set_ylabel('Price')
ax.set_title('Linear Regression Model for House Price Prediction')
plt.show()
