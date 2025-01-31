# Energy Efficient Driver Assistance System Using Deep Q-Networks

This project focuses on developing an energy-efficient driver assistance system utilizing Deep Q-Networks (DQN). The system aims to optimize driving strategies to enhance energy efficiency.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

The Energy Efficient Driver Assistance System is designed to assist drivers in adopting energy-efficient driving behaviors. By leveraging reinforcement learning techniques, specifically DQNs, the system learns optimal driving strategies that minimize energy consumption.

## Features

- Implementation of various steering controllers:
  - Pure Pursuit Controller
  - Stanley Controller
  - Model Predictive Control (MPC) Enhanced Stanley Controller
- Adaptive Cruise Control (ACC) using DQN
- Basic vehicle modeling for simulation purposes

## Installation

To set up the project, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/RohanPMore/Energy_Efficient_Driver_Assistance_System_DQN_Based_PYTHON.git
   cd Energy_Efficient_Driver_Assistance_System_DQN_Based_PYTHON
   ```

2. **Install the required dependencies:**

   Ensure you have Python installed. Then, install the necessary packages:

   ```bash
   pip install -r requirements.txt
   ```

   *(Note: As the `requirements.txt` file is not listed, please create one with all the dependencies used in your project.)*

## Usage

Detailed instructions on how to run the simulations and models:

1. **Pure Pursuit Controller:**

   ```bash
   python Pure_Pursuit_Controller.py
   ```

2. **Stanley Controller:**

   ```bash
   python Stanley_Steering_Controller.py
   ```

3. **MPC Enhanced Stanley Controller:**

   ```bash
   python MPC_Enhanced_Stanley_Steering_Controller.py
   ```

4. **Adaptive Cruise Control with DQN:**

   ```bash
   python acc_dqn_exe.py
   ```

*(Provide specific details on the expected inputs, configuration settings, and any other necessary information to run these scripts.)*

## Project Structure

An overview of the project's structure:

```
Energy_Efficient_Driver_Assistance_System_DQN_Based_PYTHON/
├── Pure_Pursuit_Controller.py
├── Stanley_Steering_Controller.py
├── MPC_Enhanced_Stanley_Steering_Controller.py
├── acc_dqn.py
├── acc_dqn_exe.py
├── basics_vehicle_model.py
├── Pure_Pursuit_Control_Wiki.md
├── Stanley_Control_Wiki.md
├── acc_dqn_wiki.md
└── README.md
```

## Contributing

Contributions are welcome! If you have suggestions or improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Deep Reinforcement Learning-Based Vehicle Energy Efficiency Autonomous Learning System](https://github.com/Luoyadan/DQN_vehicle)

---

