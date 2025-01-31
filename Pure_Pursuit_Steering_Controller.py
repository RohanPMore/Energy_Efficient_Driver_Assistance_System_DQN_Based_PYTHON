import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from beamngpy import BeamNGpy, Vehicle, Scenario, Road

class PurePursuitController:
    def __init__(self, path_file, lookahead_distance):
        self.path = pd.read_csv(path_file)  # Read the entire path
        self.lookahead_distance = lookahead_distance
        self.controller_path = []  # Initialize an empty list to store controller path

    def find_closest_point(self, current_position):
        distances = np.sqrt((self.path['x'] - current_position[0])**2 + (self.path['y'] - current_position[1])**2)
        closest_point_index = np.argmin(distances)
        return closest_point_index

    def get_lookahead_point(self, current_position, closest_point_index):
        distances = np.sqrt((self.path['x'] - current_position[0])**2 + (self.path['y'] - current_position[1])**2)
        remaining_distances = distances[closest_point_index:]
        total_distance = np.cumsum(remaining_distances)
        lookahead_index = np.argmax(total_distance > self.lookahead_distance) + closest_point_index
        lookahead_point = np.array([self.path.loc[lookahead_index, 'x'], self.path.loc[lookahead_index, 'y']])
        return lookahead_point

    def calculate_curvature(self, current_position, lookahead_point):
        x, y = current_position
        x1, y1 = lookahead_point
        distance = np.sqrt((x1 - x)**2 + (y1 - y)**2)
        if distance == 0:
            return 0
        curvature = 2 * (x1 - x) / distance**2
        return curvature

    def get_lookahead_point_with_curvature(self, current_position, closest_point_index):
        lookahead_point = self.get_lookahead_point(current_position, closest_point_index)
        curvature = self.calculate_curvature(current_position, lookahead_point)
        
        # Adjust lookahead distance based on curvature
        if curvature != 0:
            lookahead_distance_adjusted = min(self.lookahead_distance, 1 / abs(curvature))
        else:
            lookahead_distance_adjusted = self.lookahead_distance
        
        total_distance = 0
        lookahead_index = closest_point_index
        for i in range(closest_point_index, len(self.path)):
            total_distance += np.sqrt((self.path.loc[i, 'x'] - current_position[0])**2 + (self.path.loc[i, 'y'] - current_position[1])**2)
            if total_distance >= lookahead_distance_adjusted:
                lookahead_index = i
                break
        lookahead_point = np.array([self.path.loc[lookahead_index, 'x'], self.path.loc[lookahead_index, 'y']])
        return lookahead_point

    def pure_pursuit(self, current_position):
        closest_point_index = self.find_closest_point(current_position)
        lookahead_point = self.get_lookahead_point_with_curvature(current_position, closest_point_index)
        
        # Calculate steering angle using pure pursuit algorithm
        L = np.sqrt((lookahead_point[0] - current_position[0])**2 + (lookahead_point[1] - current_position[1])**2)
        x, y = current_position
        x1, y1 = lookahead_point
        alpha = np.arctan2(y1 - y, x1 - x)
        delta = np.arctan2(2 * 1.5 * np.sin(alpha), L)
        
        # Store the path followed by the controller
        self.controller_path.append(current_position)

        return delta

if __name__ == "__main__":
    path_file = "AMORiPathAndDistanceSpeedLimits.csv"
    lookahead_distance = 0.01  # You can adjust this value as needed
    controller = PurePursuitController(path_file, lookahead_distance)

    # Connect to the BeamNG server
    bng = BeamNGpy('127.0.0.1', 64535, remote=True)
    bng.open(launch=False)
    scenario = Scenario('smallgrid', 'Test')
    vehicle = Vehicle('ego_vehicle', model='renault_zoe_q90', license='PYTHON')
    scenario.add_vehicle(vehicle, pos=(0, 0, 0))

    # Start the scenario
    scenario.make(bng)
    bng.load_scenario(scenario)
    bng.start_scenario()

    vehicle.set_shift_mode('arcade')
    vehicle.poll_sensors()

    # Initialize plot
    plt.figure(figsize=(8, 6))
    plt.ion()  # Turn on interactive mode

    # Plot the original path
    plt.plot(controller.path['x'], controller.path['y'], label='Path', color='blue')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Final Path Followed by the Controller with Vehicle Position')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    # Simulation loop
    distance_covered = 0
    previous_position = None

    for i in range(len(controller.path)):
        # Extract vehicle position from state
        vehicle_position = np.array([vehicle.state['pos'][0], vehicle.state['pos'][1]])

        # Check for sudden changes in coordinates
        if abs(controller.path.loc[i, 'x'] - vehicle_position[0]) > 500 or abs(controller.path.loc[i, 'y'] - vehicle_position[1]) > 500:
            vehicle_position = np.array([controller.path.loc[i, 'x'], controller.path.loc[i, 'y']])

        if previous_position is not None:
            distance_covered += np.sqrt((vehicle_position[0] - previous_position[0])**2 + (vehicle_position[1] - previous_position[1])**2)
        
        previous_position = vehicle_position

        steering_angle = controller.pure_pursuit(vehicle_position)

        # Update vehicle position and steering angle
        vehicle.control(throttle=1, steering=steering_angle)

        # Plot vehicle positions
        print(f"Total distance covered by the vehicle: {distance_covered:.2f} units")
        plt.scatter(vehicle_position[0], vehicle_position[1], label='Vehicle Position', color='red')
        plt.pause(0.01)  # Pause to update the plot

    plt.ioff()  # Turn off interactive mode after the loop
    plt.show()  # Display the final plot

    print(f"Total distance covered by the vehicle: {distance_covered:.2f} units")
