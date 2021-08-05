from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split as tts
import matplotlib.pyplot as plt
import pandas as pd

# --------------
# Load your data
# --------------
data = pd.read_csv('data.csv').to_numpy()
X = data[:, 1:]
y = data[:, 0]
# ---------------------
# Train-vali-test split
# ---------------------
X_tv, X_test, y_tv, y_test = tts(X, y, test_size=0.25, random_state=1)
X_train, X_val, y_train, y_val = tts(X_tv, y_tv, test_size=1/3, random_state=1)

# -----------------------------------------
# Build and fit your regression model on 
# the combined training and validation data
# -----------------------------------------
linear_reg = LinearRegression() 
degree = 3
linear_reg.fit(X_tv[:, :degree], y_tv)
# Modify so that you fit a polynomial of degree 3
# Your model should be trained on the combined training and validation data

# -------------------
# Plot your test data
# -------------------

plt.scatter(X_test[:, 0], y_test, color='red', label='Test data')

# --------------
# Plot the model (Do not modify)
# --------------
x_linspace = pd.read_csv('x_linspace.csv').to_numpy()
x_model = x_linspace[:, 1]
y_model = linear_reg.predict(x_linspace[:, 1:3+1])
plt.plot(x_model, y_model, color = 'blue', label = "Model")

plt.plot(x_linspace[:, 1], x_linspace[:, 0], label = 'True', color = 'black')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Model: polynomial of degree 3')
plt.legend()
plt.ylim([-2000, 1500])
plt.savefig('plot.png')
