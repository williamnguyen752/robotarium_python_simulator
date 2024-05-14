import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.graph import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import numpy as np
import time

# Experiment Constants
iterations = 100  # Run the simulation/experiment for 1000 steps
N = 4  # Number of robots to use, this must stay 4 unless the Laplacian is changed.

waypoints = np.array([[-1, -1, 1, 1],[0.8, -0.8, -0.8, 0.8]])  # Waypoints the leader moves to.
close_enough = 0.03;  # How close the leader must get to the waypoint to move to the next one.

# Create a cyclic communication graph
L = cycle_GL(N)

# Initialize attack variables
attack_type = np.zeros(N, dtype=int)
dos_attack_steps = range(30, 60)  # Steps when DoS attack is active

# Kalman Filter-based Attack Detection
def kalman_filter_attack_detection(x, L, Q, R):
    # Kalman filter parameters
    n = x.shape[0]  # Number of states
    m = x.shape[1]  # Number of robots
    A = np.eye(n)   # State transition matrix
    B = np.eye(n)   # Control input matrix
    C = np.eye(n)   # Observation matrix

    # Adjust the dimensions of Q and R based on the number of states
    if Q.shape[0] != n:
        Q = np.eye(n) * Q[0, 0]  # Assume Q is a scalar multiple of identity matrix
    if R.shape[0] != n:
        R = np.eye(n) * R[0, 0]  # Assume R is a scalar multiple of identity matrix
    
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
    
    # Compute residuals
    residuals = np.linalg.norm(x_hat - x, axis=0)
    
    # Detect attacks based on residual thresholds
    attack_detected = np.zeros(m, dtype=bool)
    attack_detected[residuals > 0.5] = True  # Adjust the threshold as needed
    
    return attack_detected

# Secure Control Protocol
def secure_control_protocol(x, L, attack_detected):
    # Initialize control input
    u = np.zeros((2, N))
    
    # Leader-follower consensus for secure control
    for i in range(N):
        if attack_detected[i]:
            # Robot i is under attack, follow the leader
            leader_idx = np.argmin(attack_detected)  # Assume the leader is not under attack
            u[:, i] = x[:, leader_idx] - x[:, i]
        else:
            # Robot i is not under attack, use standard consensus
            neighbors = np.nonzero(L[i])[0]
            if len(neighbors) > 0:
                u[:, i] = np.sum(x[:, neighbors] - x[:, i, np.newaxis], axis=1)
    
    return u

# Instantiate the Robotarium object with these parameters
initial_conditions = np.array([[0, 0.5, 0.3, -0.1],[0.5, 0.5, 0.2, 0],[0, 0, 0, 0]])
r = robotarium.Robotarium(number_of_robots=N, show_figure=True, initial_conditions=initial_conditions, sim_in_real_time=True)

# Grab Robotarium tools to do single-integrator to unicycle conversions and collision avoidance
_, uni_to_si_states = create_si_to_uni_mapping()
si_to_uni_dyn = create_si_to_uni_dynamics()
si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()

# Initialize leader state
state = 0

# Limit maximum linear speed of any robot
magnitude_limit = 0.15

# Create gains for our formation control algorithm
formation_control_gain = 10
desired_distance = 0.3

Q = np.eye(2) # Process noise covariance
R = np.eye(2) # Measurement noise covariance

# Storage variables
poses_with_countermeasure = []
velocities_with_countermeasure = []

# Simulation loop with countermeasure
for t in range(iterations):

    # Get the most recent pose information from the Robotarium.
    x = r.get_poses()
    xi = uni_to_si_states(x)

    # Adjust Q and R based on the number of states
    n = xi.shape[0]  # Number of states
    if t == 0:
        Q = np.eye(n) * 0.1
        R = np.eye(n) * 0.5

    # Perform Kalman filter-based attack detection
    attack_detected = kalman_filter_attack_detection(xi, L, Q, R)

    # Apply secure control protocol
    dxi = secure_control_protocol(xi, L, attack_detected)

    # Keep single integrator control vectors under specified magnitude
    norms = np.linalg.norm(dxi, 2, 0)
    idxs_to_normalize = (norms > magnitude_limit)
    dxi[:, idxs_to_normalize] *= magnitude_limit / norms[idxs_to_normalize]

    # Simulate DoS attack on followers only
    if t in dos_attack_steps:
        followers_indices = np.arange(1, N)
        dxi[:, followers_indices] = np.zeros((2, N-1))

    # Use barriers and convert single-integrator to unicycle commands
    dxi = si_barrier_cert(dxi, x[:2, :])
    dxu = si_to_uni_dyn(dxi, x)

    # Set the velocities of agents 1,...,N to dxu
    r.set_velocities(np.arange(N), dxu)

    poses_with_countermeasure.append(x.copy())
    velocities_with_countermeasure.append(dxi.copy())

    # Iterate the simulation
    r.step()

# Convert poses_with_countermeasure and velocities_with_countermeasure to numpy arrays
poses_history = np.array(poses_with_countermeasure)
velocities_history = np.array(velocities_with_countermeasure)

# Call at end of script to print debug information and for your script to run on the Robotarium server properly
r.call_at_scripts_end()

# Plot the results
import matplotlib.pyplot as plt

def plot_results(poses, velocities, attack_detected):
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
    
    # Plot robot velocities
    plt.figure()
    for i in range(num_robots):
        plt.plot(velocities[:, 0, i], label=f'Robot {i+1} - Vx')
        plt.plot(velocities[:, 1, i], label=f'Robot {i+1} - Vy')
    plt.xlabel('Time')
    plt.ylabel('Velocity')
    plt.title('Robot Velocities')
    plt.legend()
    plt.grid(True)
    
    # Plot attack detection results
    plt.figure()
    for i in range(num_robots):
        if attack_detected[i]:
            plt.plot(poses[-1, i, 0], poses[-1, i, 1], 'ro', markersize=10, label=f'Robot {i+1} - Attacked')
        else:
            plt.plot(poses[-1, i, 0], poses[-1, i, 1], 'bo', markersize=8, label=f'Robot {i+1} - Normal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Attack Detection')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    plt.pause(15)  # Pause for 5 seconds

attack_detection_results = kalman_filter_attack_detection(poses_history[-1], L, Q, R)
plot_results(poses_history, velocities_history, attack_detection_results)