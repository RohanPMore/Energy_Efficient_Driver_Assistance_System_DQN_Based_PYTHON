import numpy as np

# Constants
g = 9.81  # Acceleration due to gravity (m/s^2)
rho_air = 1.2  # Air density (kg/m^3)
Cd = 0.3  # Drag coefficient
A_cross_section = 2.5  # Cross-sectional area of the vehicle (m^2)

# Vehicle parameters
mass_vehicle = 1500  # Mass of the vehicle (kg)
rolling_resistance_coef = 0.01  # Rolling resistance coefficient

def calculate_resistive_forces(velocity, gradient):
    # Calculate resistive forces: air drag, rolling resistance, gradient
    air_drag = 0.5 * rho_air * Cd * A_cross_section * velocity**2
    rolling_resistance = rolling_resistance_coef * mass_vehicle * g * np.cos(np.arctan(gradient))
    gradient_resistance = mass_vehicle * g * np.sin(np.arctan(gradient))
    
    return air_drag + rolling_resistance + gradient_resistance

def calculate_acceleration_force(acceleration):
    # Calculate force required for acceleration
    acceleration_force = mass_vehicle * acceleration
    return acceleration_force

def calculate_fuel_consumption(velocity, time, gradient, acceleration):
    # Calculate fuel consumption based on resistive forces, acceleration force, and time
    resistive_forces = calculate_resistive_forces(velocity, gradient)
    acceleration_force = calculate_acceleration_force(acceleration)
    
    total_force = resistive_forces + acceleration_force
    energy_required = total_force * velocity * time  # Energy in Joules
    
    fuel_consumption_rate = 0.05  # Arbitrary fuel consumption rate (adjust as needed)
    fuel_consumption = energy_required / (fuel_consumption_rate * 3600 * 1000)  # Convert J to liters
    
    return fuel_consumption

if __name__ == "__main__":
    # Example scenarios
    velocity = 20  # Initial velocity (m/s)

    # Scenario 1: Straight road with constant speed
    time_straight = 600  # Time spent on straight road (seconds)
    gradient_straight = 0  # No gradient on straight road
    acceleration_straight = 0  # No acceleration on straight road
    fuel_consumption_straight = calculate_fuel_consumption(velocity, time_straight, gradient_straight, acceleration_straight)
    print(f"Fuel consumption on straight road: {fuel_consumption_straight:.2f} liters")

    # Scenario 2: Climbing a hill
    time_climbing_hill = 300  # Time spent climbing the hill (seconds)
    gradient_hill = 0.1  # Gradient of the hill (tan(theta))
    acceleration_hill = 1.0  # Acceleration while climbing (m/s^2)
    fuel_consumption_hill = calculate_fuel_consumption(velocity, time_climbing_hill, gradient_hill, acceleration_hill)
    print(f"Fuel consumption climbing the hill: {fuel_consumption_hill:.2f} liters")

    # Scenario 3: Descending a hill
    time_descending_hill = 200  # Time spent descending the hill (seconds)
    gradient_downhill = -0.1  # Negative gradient for descending
    acceleration_downhill = -0.5  # Deceleration while descending (m/s^2)
    fuel_consumption_downhill = calculate_fuel_consumption(velocity, time_descending_hill, gradient_downhill, acceleration_downhill)
    print(f"Fuel consumption descending the hill: {fuel_consumption_downhill:.2f} liters")

    # Scenario 4: Overtaking a vehicle
    time_overtaking = 120  # Time spent overtaking vehicles (seconds)
    gradient_overtaking = 0  # Assume flat road for overtaking
    acceleration_overtaking = 2.0  # Acceleration while overtaking (m/s^2)
    fuel_consumption_overtaking = calculate_fuel_consumption(velocity, time_overtaking, gradient_overtaking, acceleration_overtaking)
    print(f"Fuel consumption while overtaking: {fuel_consumption_overtaking:.2f} liters")
