import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.graph import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import numpy as np
import time

# Simulation Parameters
N = 4  # Number of robots
T = 1000  # Total simulation steps
dos_attack_steps = range(30, 60)  # Steps when DoS attack is active
L = cycle_GL(N)  # Cycle graph Laplacian

# Create the Robotarium object
r = robotarium.Robotarium(number_of_robots=N, show_figure=True, sim_in_real_time=True)

# Create single-integrator position controller and barrier certificates
si_controller = create_si_position_controller()
# Grab Robotarium tools to do simgle-integrator to unicycle conversions and collision avoidance
# Single-integrator -> unicycle dynamics mapping
si_to_uni_dyn = create_si_to_uni_dynamics()
si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()

# Storage variables
poses_without_countermeasure = []

# Helper function to compute consensus error
def compute_consensus_error(poses):
    errors = np.linalg.norm(poses - np.mean(poses, axis=1, keepdims=True), axis=0)
    return np.mean(errors)

# Simulation loop without countermeasure
for t in range(T):
    # Get poses
    poses = r.get_poses()
    velocities = np.zeros((2, N))

    # Simulate DoS attack by zeroing out control inputs
    if t not in dos_attack_steps:
        for i in range(N):
            neighbors = topological_neighbors(L, i)
            if len(neighbors) > 0:
                desired_pos = np.mean(poses[:2, neighbors], axis=1, keepdims=True)
                velocities[:, i:i+1] = si_controller(poses[:2, i:i+1], desired_pos)

    # Apply barrier certificates and map to unicycle dynamics
    si_velocities = si_barrier_cert(velocities, poses[:2])  # Pass only x and y coordinates
    dxu = si_to_uni_dyn(si_velocities, poses)  
    r.set_velocities(np.arange(N), dxu)

    poses_without_countermeasure.append(poses.copy())
    r.step()

# Convert pose lists to numpy arrays
poses_without_countermeasure = np.array(poses_without_countermeasure)

# Compute consensus errors
errors_without_countermeasure = [compute_consensus_error(poses) for poses in poses_without_countermeasure]

print(f"Average consensus error without countermeasure: {np.mean(errors_without_countermeasure):.3f}")

r.call_at_scripts_end()