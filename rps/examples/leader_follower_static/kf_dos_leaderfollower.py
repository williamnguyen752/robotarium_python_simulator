import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.graph import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import numpy as np
import time

###
def create_custom_graph(num_robots, topology):
    # Create a custom communication graph based on the desired topology
    # Input: num_robots - number of robots in the system
    #        topology - string representing the desired topology
    # Output: L - Laplacian matrix representing the desired topology
    # Note: This function should return the Laplacian matrix representing the desired topology
    #       The Laplacian matrix should be a numpy array of shape (num_robots, num_robots)
    #       The Laplacian matrix should be symmetric and have zeros on the diagonal
    #       The Laplacian matrix should have negative values on the off-diagonal elements
    #       The Laplacian matrix should have positive values on the diagonal elements
    #       The Laplacian matrix should have the following properties based on the desired topology:
    #           - Complete graph: All off-diagonal elements are -1
    #           - Cycle graph: Adjacent elements are -1, and the first and last elements are -1
    #           - Star graph: All off-diagonal elements are -1, except the first element which is num_robots-1
    #           - Line graph: Adjacent elements are -1
    #           - Custom graph: The user can define the desired topology

    # Initialize the Laplacian matrix
    L = np.zeros((num_robots, num_robots))
    # Return the adjacency matrix or Laplacian matrix representing the graph
    if topology == 'complete':
        L = -np.ones((num_robots, num_robots))
        np.fill_diagonal(L, num_robots-1)
    elif topology == 'cycle':
        L = -np.eye(num_robots) - np.eye(num_robots, k=1)
        L[0, -1] = -1
        L[-1, 0] = -1
    elif topology == 'star':
        L = -np.ones((num_robots, num_robots))
        np.fill_diagonal(L, 0)
        L[0, :] = -1
        L[:, 0] = -1
        np.fill_diagonal(L, num_robots-1)
    elif topology == 'line':
        L = -np.eye(num_robots) - np.eye(num_robots, k=1)
    elif topology == 'custom':
        # Define the custom graph
        L = np.array([[0, -1, 0, 0],
                      [-1, 0, -1, 0],
                      [0, -1, 0, -1],
                      [0, 0, -1, 0]])
    return L
###
# Modify the existing code to use the custom graph
# L = create_custom_graph(N, topology)
###

# Experiment Constants
iterations = 1000  # Run the simulation/experiment for 1000 steps
N = 4  # Number of robots to use, this must stay 4 unless the Laplacian is changed.

waypoints = np.array([[-1, -1, 1, 1],[0.8, -0.8, -0.8, 0.8]])  # Waypoints the leader moves to.
close_enough = 0.03;  # How close the leader must get to the waypoint to move to the next one.

# Create the desired Laplacian
followers = -completeGL(N-1)
L = create_custom_graph(N, 'custom')
# L = np.zeros((N, N))
# L[1:N,1:N] = followers
# L[1,1] = L[1,1] + 1
# L[1,0] = -1

# For computational/memory reasons, initialize the velocity vector
dxi = np.zeros((2, N))

# Initialize leader state
state = 0

# Limit maximum linear speed of any robot
magnitude_limit = 0.15

# Create gains for our formation control algorithm
formation_control_gain = 10
desired_distance = 0.3

# Initial Conditions to Avoid Barrier Use in the Beginning.
initial_conditions = np.array([[0, 0.5, 0.3, -0.1],[0.5, 0.5, 0.2, 0],[0, 0, 0, 0]])

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

    # Initialize velocity vector
    dxi = np.zeros((2, N))

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

    # Iterate the simulation
    r.step()

# Convert pose lists to numpy arrays
poses_with_countermeasure = np.array(poses_with_countermeasure)

# Compute consensus errors
errors_with_countermeasure = [compute_consensus_error(poses) for poses in poses_with_countermeasure]

print(f"Average consensus error with countermeasure: {np.mean(errors_with_countermeasure):.3f}")

# Call at end of script to print debug information and for your script to run on the Robotarium server properly
r.call_at_scripts_end()