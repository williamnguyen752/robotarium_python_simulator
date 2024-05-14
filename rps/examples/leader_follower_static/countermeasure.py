import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.graph import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import numpy as np
import time

# Experiment Constants
iterations = 1000  # Run the simulation/experiment for 1000 steps
N = 4  # Number of robots to use, this must stay 4 unless the Laplacian is changed.

waypoints = np.array([[-1, -1, 1, 1],[0.8, -0.8, -0.8, 0.8]])  # Waypoints the leader moves to.
close_enough = 0.03;  # How close the leader must get to the waypoint to move to the next one.

# Create the desired Laplacian
followers = -completeGL(N-1)
### Communication Graph
def create_custom_graph(num_robots, topology):
    # Create a custom communication graph based on the desired topology
    if topology == 'ring':
        # Create a ring graph
        L = np.zeros((num_robots, num_robots))
        for i in range(num_robots):
            L[i, (i+1)%num_robots] = -1
            L[i, (i-1)%num_robots] = -1
            L[i, i] = 2
    elif topology == 'line':
        # Create a line graph
        L = np.zeros((num_robots, num_robots))
        for i in range(num_robots):
            if i > 0:
                L[i, i-1] = -1
            if i < num_robots-1:
                L[i, i+1] = -1
            L[i, i] = np.sum(np.abs(L[i]))
    else:
        raise ValueError("Unsupported topology. Please choose 'ring' or 'line'.")
    
    return L

# Modify the existing code to use the custom graph
topology = 'ring'  # Choose the desired topology ('ring' or 'line')
L = create_custom_graph(N, topology)
### end of communication graph

attack_type = np.zeros(N, dtype=int)
### detect attack
def detect_attacks(x_estimated, x_measured, threshold):
    # Compute residuals
    residuals = np.linalg.norm(x_estimated - x_measured, axis=0)
    
    # Detect attacks based on residual thresholds
    attack_type = np.zeros(N, dtype=int)
    attack_type[residuals > threshold] = 1  # Deception attack
    attack_type[np.isnan(residuals)] = 2    # DoS attack
    
    return attack_type

# Integrate the attack detection mechanism with the resilient switching controller
threshold = 0.5  # Set the threshold for attack detection
# attack_type = detect_attacks(x_hat, x, threshold)
### end of detect attack

### switching controller
def resilient_switching_controller(x, L, attack_type, weights, leader_idx, follower_idx, waypoints, state):
    # Initialize control input
    u = np.zeros((2, N))
    
    # Leader controller
    leader_velocity = leader_controller(x[:, leader_idx].reshape((2, 1)), waypoints[:, state].reshape((2, 1)))
    u[:, leader_idx:leader_idx+1] = leader_velocity
    
    # Follower controllers
    for i in follower_idx:
        if attack_type[i] == 0:  # No attack
            neighbors = np.nonzero(L[i])[0]
            u[:, i:i+1] = weights[i] * np.sum(x[:, neighbors] - x[:, i].reshape((2, 1)), axis=1).reshape((2, 1))
        elif attack_type[i] == 1:  # Deception attack
            neighbors = np.nonzero(L[i])[0]
            u[:, i:i+1] = weights[i] * np.sum(x[:, neighbors] - x[:, i].reshape((2, 1)), axis=1).reshape((2, 1))
        elif attack_type[i] == 2:  # DoS attack
            follower_velocity = follower_controller(x[:, i].reshape((3, 1)), x[:, leader_idx].reshape((3, 1)))
            u[:, i:i+1] = follower_velocity
    
    return u
# Integrate the resilient switching controller with the existing code
weights = np.ones(N)  # Set the initial weights for consensus control
# Usage: dxi = resilient_switching_controller(xi, L, attack_type, weights)
### end of switching controller

### Kalman Filter-based Consensus Controller:
def kalman_filter_consensus(x, L, Q, R, attack_type, weights, leader_idx, follower_idx, waypoints, state):
    # Kalman filter parameters
    n = x.shape[0]  # Number of states
    m = x.shape[1]  # Number of robots
    A = np.eye(n)   # State transition matrix
    B = np.eye(n)   # Control input matrix
    C = np.eye(n)   # Observation matrix
    
    # Initialize Kalman filter variables
    x_hat = np.zeros_like(x)
    P = np.eye(n)
    
    # Kalman filter update step
    for i in range(m):
        neighbors = np.nonzero(L[i])[0]
        if len(neighbors) > 0:
            z = np.mean(x[:, neighbors], axis=1)
            y = z - C.dot(x_hat[:, i])
            S = C.dot(P).dot(C.T) + R
            K = P.dot(C.T).dot(np.linalg.inv(S))
            x_hat[:, i] += K.dot(y)
            P = (np.eye(n) - K.dot(C)).dot(P)

    # Detect attacks
    attack_type = detect_attacks(x_hat, x, threshold)
    
    # Consensus control law
    u = -np.dot(np.kron(L, np.eye(n)), x_hat.reshape((n*m, 1)))
    # Resilient switching controller
    # u = resilient_switching_controller(x_hat, L, attack_type, weights)
    # u = resilient_switching_controller(x_hat, L, attack_type, weights, leader_idx, follower_idx, waypoints, state)
    
    # Kalman filter prediction step
    x_hat = A.dot(x_hat) + B.dot(u.reshape((n, m)))
    P = A.dot(P).dot(A.T) + Q
    
    return u.reshape((n, m))
### End of Kalman Filter-based Consensus Controller
# Integrate the Kalman filter-based consensus controller with the existing code
Q = np.eye(2) * 0.1  # Process noise covariance
R = np.eye(2) * 0.5  # Measurement noise covariance
# dxi = kalman_filter_consensus(xi, L, Q, R)
###

# For computational/memory reasons, initialize the velocity vector
# dxi = np.zeros((2, N))

# Initialize leader state
state = 0

# Limit maximum linear speed of any robot
magnitude_limit = 0.15

# Create gains for our formation control algorithm
formation_control_gain = 10
desired_distance = 0.3

# Initial Conditions to Avoid Barrier Use in the Beginning.
initial_conditions = np.array([[0, 0.5, 0.3, -0.1],[0.5, 0.5, 0.2, 0],[0, 0, 0, 0]])

### barrier certificates
def extended_barrier_certificates(dxi, x, safety_radius):
    # Compute pairwise distances between robots
    diff = x[:2, :, np.newaxis] - x[:2, np.newaxis, :]
    dist = np.sqrt(np.sum(diff**2, axis=0))
    
    # Compute the barrier function
    h = dist**2 - safety_radius**2
    
    # Compute the barrier constraint
    A = 2 * diff[:2, :]
    b = np.sum(diff * dxi, axis=0)
    
    # Solve the quadratic program
    u = qp_solve(dxi.flatten(), A.flatten(), b.flatten(), solver='cvxopt')
    
    return u.reshape(2, -1)

# Integrate the extended barrier certificates with the existing code
safety_radius = 0.2  # Set the safety radius for collision avoidance
# Usage: dxi = extended_barrier_certificates(dxi, xi, safety_radius)
### end barrier certificates

# Instantiate the Robotarium object with these parameters
r = robotarium.Robotarium(number_of_robots=N, show_figure=True, initial_conditions=initial_conditions, sim_in_real_time=True)

# Grab Robotarium tools to do single-integrator to unicycle conversions and collision avoidance
_, uni_to_si_states = create_si_to_uni_mapping()
si_to_uni_dyn = create_si_to_uni_dynamics()
si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()
leader_controller = create_si_position_controller(velocity_magnitude_limit=0.1)

# Simulation parameters
dos_attack_steps = range(300, 600)  # Steps when DoS attack is active

# Storage variables
poses_with_countermeasure = []
velocities_with_countermeasure = []  # Initialize an empty list for velocities

# Helper function to compute consensus error
def compute_consensus_error(poses):
    errors = np.linalg.norm(poses - np.mean(poses, axis=1, keepdims=True), axis=0)
    return np.mean(errors)

# Assign leader and followers for countermeasure
leader_index = 0
follower_indices = np.arange(1, N)

# Simulation loop with countermeasure
for t in range(iterations):

    # Get the most recent pose information from the Robotarium.
    x = r.get_poses()
    xi = uni_to_si_states(x)

    # # Initialize velocity vector
    # dxi = np.zeros((2, N))
    # Integrate the Kalman filter-based consensus controller
    # dxi = kalman_filter_consensus(xi, L, Q, R)
    dxi = kalman_filter_consensus(xi, L, Q, R, attack_type, weights, leader_index, follower_indices, waypoints, state)
    # Apply extended barrier certificates
    # dxi = extended_barrier_certificates(dxi, xi, safety_radius)


    # Followers
    for i in follower_indices:
        # Zero velocities and get the topological neighbors of agent i
        neighbors = topological_neighbors(L, int(i))  # Convert i to integer

        for j in neighbors:
            dxi[:, [i]] += formation_control_gain * (np.power(np.linalg.norm(x[:2, [j]] - x[:2, [i]]), 2) - np.power(desired_distance, 2))*(x[:2, [j]]-x[:2, [i]])

    # Leader
    waypoint = waypoints[:, state].reshape((2, 1))

    dxi[:, [leader_index]] = leader_controller(x[:2, [leader_index]], waypoint)

    if np.linalg.norm(x[:2, [leader_index]] - waypoint) < close_enough:
        state = (state + 1)%4

    # Keep single integrator control vectors under specified magnitude
    norms = np.linalg.norm(dxi, 2, 0)
    idxs_to_normalize = (norms > magnitude_limit)
    dxi[:, idxs_to_normalize] *= magnitude_limit / norms[idxs_to_normalize]

    # Simulate DoS attack on followers only
    if t in dos_attack_steps:
        dxi[:, follower_indices] = np.zeros((2, N-1))

    # Use barriers and convert single-integrator to unicycle commands
    dxi = si_barrier_cert(dxi, x[:2, :])
    dxu = si_to_uni_dyn(dxi, x)

    # Set the velocities of agents 1,...,N to dxu
    r.set_velocities(np.arange(N), dxu)

    poses_with_countermeasure.append(x.copy())
    velocities_with_countermeasure.append(dxi.copy())  # Append velocities to the list

    # Iterate the simulation
    r.step()

# Convert pose lists to numpy arrays
poses_with_countermeasure_arr = np.array(poses_with_countermeasure)

# Compute consensus errors
errors_with_countermeasure = [compute_consensus_error(poses) for poses in poses_with_countermeasure_arr]

print(f"Average consensus error with countermeasure: {np.mean(errors_with_countermeasure):.3f}")

import matplotlib.pyplot as plt

def plot_results(poses, velocities, communication_graph, attack_detection_results):
    num_steps, num_robots, _ = poses.shape
    
    # Plot robot positions
    plt.figure()
    for i in range(num_robots):
        plt.plot(poses[:, i, 0], poses[:, i, 1], label=f'Robot {i+1}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Robot Trajectories')
    plt.legend()
    plt.grid(True)
    
    # Check if velocities array is available and has the expected shape
    if velocities is not None:
        if velocities.shape == (num_steps, 2, num_robots):
            velocities_reshaped = np.transpose(velocities, (0, 2, 1))
        elif velocities.shape == (2, num_robots, num_steps):
            velocities_reshaped = np.transpose(velocities, (2, 1, 0))
        else:
            print("Warning: Velocities array has an unexpected shape. Skipping velocity plot.")
            velocities_reshaped = None
        
        # Plot robot velocities if available and has the expected shape
        if velocities_reshaped is not None:
            plt.figure()
            for i in range(num_robots):
                plt.plot(velocities_reshaped[:, i, 0], label=f'Robot {i+1} - Vx')
                plt.plot(velocities_reshaped[:, i, 1], label=f'Robot {i+1} - Vy')
            plt.xlabel('Time')
            plt.ylabel('Velocity')
            plt.title('Robot Velocities')
            plt.legend()
            plt.grid(True)
    
    # Plot communication graph
    plt.figure()
    for i in range(num_robots):
        for j in range(num_robots):
            if communication_graph[i, j] != 0:
                plt.plot([poses[-1, i, 0], poses[-1, j, 0]], 
                         [poses[-1, i, 1], poses[-1, j, 1]], 'k--')
    for i in range(num_robots):
        if i < attack_detection_results.shape[0]:
            if attack_detection_results[i] == 1:
                plt.plot(poses[-1, i, 0], poses[-1, i, 1], 'ro', markersize=10)
            elif attack_detection_results[i] == 2:
                plt.plot(poses[-1, i, 0], poses[-1, i, 1], 'rx', markersize=10)
            else:
                plt.plot(poses[-1, i, 0], poses[-1, i, 1], 'bo', markersize=8)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Communication Graph')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    plt.pause(15)  # Pause for 5 seconds

# Integrate plotting and validation with the existing code
poses_history = np.array(poses_with_countermeasure)
velocities_history = np.array(velocities_with_countermeasure)
# usage plotting
print("Velocities shape:", velocities_history.shape)
plot_results(poses_history, velocities_history, L, attack_type)


# Call at end of script to print debug information and for your script to run on the Robotarium server properly
r.call_at_scripts_end()