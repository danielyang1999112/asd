import pandas as pd
import numpy as np 

def grad_beta0(x, y, beta0, beta1):
    return np.sum(beta0 + beta1 * x - y)

def grad_beta1(x, y, beta0, beta1):
    return np.sum(x *(beta0 + beta1 * x - y))

baton_rouge = pd.read_csv('BatonRouge.csv')

x = baton_rouge['SQFT'].to_numpy()
y = baton_rouge['Price'].to_numpy()

# Initialisation
beta0 = 0
beta1 = 0

alpha0 = 1e-3
alpha1 = 1e-10

max_iterations = 300
tol = 0.001

print('Initial: {:.4f}, {:.4f}'.format(beta0, beta1))

# Run gradient descent
for i in range(max_iterations):
    beta0_new = beta0 - alpha0 * grad_beta0(x, y, beta0, beta1)
    beta1_new = beta1 - alpha1 * grad_beta1(x, y, beta0, beta1)
    
    if np.abs(beta0 - beta0_new) < tol and np.abs(beta1 - beta1_new) < tol:
        break

    beta0 = beta0_new
    beta1 = beta1_new
    
    print('{}: {:.4f}, {:.4f}'.format(i, beta0, beta1))
