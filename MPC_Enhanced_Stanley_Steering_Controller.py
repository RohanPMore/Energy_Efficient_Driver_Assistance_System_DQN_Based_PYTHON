import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import cvxpy as cp

# Function to calculate the nearest point on the path
def find_nearest_point(x, y, path_x, path_y):
    distances = np.sqrt((path_x - x)**2 + (path_y - y)**2)
    min_idx = np.argmin(distances)
    return min_idx, distances[min_idx]

# Enhanced Stanley Controller with Look-ahead
def stanley_control(x, y, yaw, v, path_x, path_y, path_yaw, k=1.0, look_ahead=5):
    nearest_idx, _ = find_nearest_point(x, y, path_x, path_y)
    
    # Look-ahead implementation
    look_ahead_idx = min(nearest_idx + look_ahead, len(path_x) - 1)
    target_x = path_x[look_ahead_idx]
    target_y = path_y[look_ahead_idx]
    target_yaw = path_yaw[look_ahead_idx]
    
    # Calculate heading error
    heading_error = target_yaw - yaw
    heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))  # Normalize angle

    # Calculate crosstrack error
    crosstrack_error = np.linalg.norm([x - target_x, y - target_y])
    sign = np.sign(np.dot([np.cos(yaw), np.sin(yaw)], [target_x - x, target_y - y]))
    crosstrack_error *= sign

    # Adaptive gain based on velocity
    k_adaptive = k * v

    # Stanley control law with feedforward term
    delta = heading_error + np.arctan2(k_adaptive * crosstrack_error, v) + 0.1 * target_yaw

    return delta

# MPC Controller
def mpc_control(x, y, yaw, v, path_x, path_y, path_yaw, dt=0.1, horizon=10):
    x_ref = np.zeros(horizon)
    y_ref = np.zeros(horizon)
    yaw_ref = np.zeros(horizon)
    
    nearest_idx, _ = find_nearest_point(x, y, path_x, path_y)
    for i in range(horizon):
        idx = min(nearest_idx + i, len(path_x) - 1)
        x_ref[i] = path_x[idx]
        y_ref[i] = path_y[idx]
        yaw_ref[i] = path_yaw[idx]

    # Variables
    delta = cp.Variable(horizon)
    state = cp.Variable((3, horizon + 1))

    # Cost function
    cost = 0
    for t in range(horizon):
        cost += cp.sum_squares(state[0, t] - x_ref[t])
        cost += cp.sum_squares(state[1, t] - y_ref[t])
        cost += cp.sum_squares(state[2, t] - yaw_ref[t])
        cost += cp.sum_squares(delta[t])

    # Constraints
    constraints = []
    constraints.append(state[:, 0] == [x, y, yaw])
    for t in range(horizon):
        # Linearize the state transitions using first-order approximation
        state_x_next = state[0, t] + v * np.cos(yaw_ref[t]) * dt
        state_y_next = state[1, t] + v * np.sin(yaw_ref[t]) * dt
        state_yaw_next = state[2, t] + v / L * delta[t] * dt
        constraints.append(state[0, t + 1] == state_x_next)
        constraints.append(state[1, t + 1] == state_y_next)
        constraints.append(state[2, t + 1] == state_yaw_next)

    # Solve the problem
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()

    return delta.value[0]

# Function to update vehicle state
def update_state(x, y, yaw, v, delta, dt):
    x += v * np.cos(yaw) * dt
    y += v * np.sin(yaw) * dt
    yaw += v / L * np.tan(delta) * dt
    yaw = np.arctan2(np.sin(yaw), np.cos(yaw))  # Normalize angle
    return x, y, yaw

# Function to detect extreme changes in coordinates
def detect_extreme_change(current_x, current_y, new_x, new_y, threshold=200):
    distance = np.sqrt((new_x - current_x)**2 + (new_y - current_y)**2)
    return distance > threshold

# Read the CSV data
file_path = 'AMORiPathAndDistanceSpeedLimits.csv'
data = pd.read_csv(file_path)

# Smooth the path using Savitzky-Golay filter
window_size = 51  # Window size for the filter, must be odd
poly_order = 3    # Polynomial order for the filter

data['y_smooth'] = savgol_filter(data['y'], window_size, poly_order)
data['x_smooth'] = savgol_filter(data['x'], window_size, poly_order)

# Convert the smoothed path to numpy arrays
path_x = data['x_smooth'].to_numpy()
path_y = data['y_smooth'].to_numpy()

# Calculate path heading
path_yaw = np.arctan2(np.diff(path_y, append=path_y[-1]), np.diff(path_x, append=path_x[-1]))

# Simulation parameters
dt = 0.05  # Time step
L = 2.5    # Wheelbase of the vehicle
v = 5.0    # Velocity of the vehicle

# Initial state of the vehicle
x = path_x[0]
y = path_y[0]
yaw = path_yaw[0]

# Lists to store the vehicle's path
actual_x = [x]
actual_y = [y]

# Variable to accumulate distance
distance_covered = 0

# Simulation loop
for i in range(1, len(path_x)):
    # Check for extreme change in coordinates
    if detect_extreme_change(x, y, path_x[i], path_y[i]):
        x = path_x[i]
        y = path_y[i]
        yaw = path_yaw[i]
    else:
        delta = mpc_control(x, y, yaw, v, path_x, path_y, path_yaw, dt)
        x_new, y_new, yaw_new = update_state(x, y, yaw, v, delta, dt)
        
        # Calculate the distance covered in this step
        step_distance = np.sqrt((x_new - x)**2 + (y_new - y)**2)
        distance_covered += step_distance
        
        x, y, yaw = x_new, y_new, yaw_new
    
    actual_x.append(x)
    actual_y.append(y)

# Plot the paths
plt.figure(figsize=(10, 6))

# Original smoothed path
plt.plot(path_x, path_y, label='Smoothed Path')

# Vehicle's actual path
plt.plot(actual_x, actual_y, label='Traversed Path', linestyle='--')

# Adding labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D Path with MPC-Enhanced Stanley Controller')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

# Print the total distance covered
print(f"Total distance covered by the vehicle: {distance_covered:.2f} units")
