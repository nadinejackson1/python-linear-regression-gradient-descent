import numpy as np

# Step 1: Define the cost function (Mean Squared Error)
# Compute the mean squared error (MSE) for a given set of data points (x, y) and line parameters (m, b)

def cost_function(m, b, x, y):
    n = len(x)
    total_error = np.sum((y - (m * x + b))**2)
    return total_error / n
# Step 2: Compute the gradients with respect to m and b

def gradients(m, b, x, y):
    n = len(x)
    dm = -(2/n) * np.sum(x * (y - (m * x + b)))
    db = -(2/n) * np.sum(y - (m * x + b))
    return dm, db
# Step 3: Update the parameters (m and b) using gradient descent

def gradient_descent(m, b, x, y, learning_rate, iterations):
    for _ in range(iterations):
        dm, db = gradients(m, b, x, y)
        m -= learning_rate * dm
        b -= learning_rate * db
    return m, b
# Step 4: Test the algorithm on a dataset
# Assuming x and y are NumPy arrays containing the data points

x = np.array([1, 2, 3, 4, 5])
y = np.array([3, 4, 5, 6, 7])
# Initialize m and b to any values

m_init, b_init = 0, 0
# Set the learning rate and number of iterations

learning_rate = 0.01
iterations = 1000
# Perform gradient descent to find the best fit line

m_optimal, b_optimal = gradient_descent(m_init, b_init, x, y, learning_rate, iterations)
# Print the results

print(f"Optimal m: {m_optimal}, Optimal b: {b_optimal}")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Step 6: Visualize the cost function

def plot_cost_function(x, y, m_optimal, b_optimal):
    m_values = np.linspace(m_optimal - 2, m_optimal + 2, 100)
    b_values = np.linspace(b_optimal - 2, b_optimal + 2, 100)
    
    M, B = np.meshgrid(m_values, b_values)
    cost = np.array([cost_function(m, b, x, y) for m, b in zip(np.ravel(M), np.ravel(B))])
    Cost = cost.reshape(M.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot_surface(M, B, Cost, cmap='viridis', alpha=0.8)
    ax.scatter(m_optimal, b_optimal, cost_function(m_optimal, b_optimal, x, y), c='red', marker='o', s=100)
    
    ax.set_xlabel('m')
    ax.set_ylabel('b')
    ax.set_zlabel('Cost')

    plt.show()


# Call the function to plot the cost function

plot_cost_function(x, y, m_optimal, b_optimal)