from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse 
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
X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.25, random_state = 1)
X_train, X_vali, y_train, y_vali = train_test_split(X_tv, y_tv, test_size=1/3, random_state = 1)

mse_train = []
mse_vali = []

max_deg = 20
degrees = range(1, max_deg+1)

for deg in degrees:

    # -----------------------------------
    # Build and fit your regression model
    # -----------------------------------
    linear_reg = LinearRegression() 
    linear_reg.fit(X_train[:, :deg], y_train)
    # Modify so that you fit a polynomial of degree 'deg'
    # ----------------------------------
    # Predict with your regression model
    # ----------------------------------
    pred_train = linear_reg.predict(X_train[:, :deg])
    pred_vali = linear_reg.predict(X_vali[:, :deg])
    # -----------------
    # Calculate and save the MSE
    # -----------------
    mse_train.append(mse(pred_train, y_train))
    mse_vali.append(mse(pred_vali, y_vali))
    # Modify your code so you append the MSE results to the respective arrays

# ------------------------
# Plot the holdout results (Do not modify)
# ------------------------
plt.plot(degrees, mse_train, color = 'green', label = 'Training data')
plt.plot(degrees, mse_vali, color = 'orange', label = 'Validation data')
plt.xlabel('Polynomial degree')
plt.ylabel('Mean squared error')
plt.title('Holdout')
plt.xticks(degrees)
plt.yscale('log')
plt.ylim([5e3, 5e11])
plt.legend()
plt.savefig('plot.png')
