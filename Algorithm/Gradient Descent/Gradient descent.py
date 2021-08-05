import matplotlib.pyplot as plt
import numpy as np
def f(x):
    y = 2*pow(x,4) - pow(x,3) - 10*pow(x,2) + 6*x +12
    return y

def grad(x):
    yy = 8*pow(x,3) - 3*pow(x,2) - (20*pow(x,1)) + 6
    return yy
# Initialisation
x = 0
alpha = 0.01
max_iterations = 25
# Save values
xs = []
ys = []
xs.append(x)
ys.append(f(x))
# Run gradient descent
for i in range(max_iterations):
    x -= alpha * grad(x)
    xs.append(x)
    ys.append(f(x))
# Plotting
func_x = np.linspace(-2.5, 2.5, 100)
plt.plot(func_x, f(func_x), color = 'gray', alpha = 0.2)
plt.scatter(xs, ys, s = 30, c = range(len(xs)), cmap = 'binary', edgecolor = 'black', linewidth = 0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('plot.png')
