import numpy as np
from sklearn.linear_model import LinearRegression

# Simple dataset
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

model = LinearRegression()
model.fit(X, y)

prediction = model.predict([[6]])

print("Model trained successfully!")
print("Prediction for input 6:", prediction[0])
